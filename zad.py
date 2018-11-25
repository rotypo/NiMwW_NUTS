import numpy as np
import sympy as sp
import pymc3 as pm
from sympy.abc import i,x
from matplotlib import pyplot as ppl
import theano.tensor as tt
import seaborn as sns

K = lambda x: sp.exp(-(x**2)/2)/sp.sqrt(2*sp.pi)

n = 10

r29 = np.loadtxt("rok2009.txt")

Ldwnn = np.random.choice(r29[:,0],n)
Ln = np.random.choice(r29[:,1],n)

h = 1.06*np.std(Ldwnn)*n**(-1/5)

Ldwn = sp.Array(Ldwnn)

h = sp.symbols('h')

f = lambda x, h: 1/(n*h)*sp.Sum(K((x - Ldwn[i-1])/h), (i, 1, n)).doit()

#ran = (x, np.min(Ldwn)-1/h, np.max(Ldwn)+1/h)
#
#p = sp.plot(f(x), ran, show=False)
#
#for xi in range(n):
#    p.extend(sp.plot(K((x-Ldwn[xi])/h)/(n*h), ran, line_color='r', show=False))
#
#ppl.show(block=False)


from numpy import exp
exec(open("prior2008LDWN.m").read())

f_lbd = sp.lambdify((x, h), f(x, h))


def logp_theta(value):
    return np.log(prior2008LDWN(value)).sum()

def logp_est(value, h):
    return np.log(f_lbd(value, h)).sum()

with pm.Model() as model:
    theta = pm.DensityDist('theta', logp_theta, testval=np.mean(Ldwnn))
    h = 1.06*theta.std()*n**(-1/5) + 1e-10
    like = pm.DensityDist('y_est', logp_est, observed={'value': Ldwnn, 'h': h})
    trace = pm.sample(100000, nuts_kwargs={'target_accept': 0.9})

pl = sns.distplot(trace.get_values('theta'))
xx = pl.lines[0].get_xdata()
pl.plot(xx, prior2008LDWN(xx))
pl.legend(('posterior', 'posterior', 'prior'))
ppl.show(block=False)
