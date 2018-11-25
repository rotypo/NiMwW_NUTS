import numpy as np
import sympy as sp
import pymc3 as pm
from sympy.abc import i,x
from matplotlib import pyplot as ppl
import seaborn as sns

K = lambda x: sp.exp(-(x**2)/2)/sp.sqrt(2*sp.pi)

n = 10

r29 = np.loadtxt("rok2009.txt")

Ldwnn = np.random.choice(r29[:,0],n)
Ln = np.random.choice(r29[:,1],n)

h = 1.06*np.std(Ldwnn)*n**(-1/5)

Ldwn = sp.Array(Ldwnn)

f = lambda x: 1/(n*h)*sp.Sum(K((x - Ldwn[i-1])/h), (i, 1, n)).doit()

ran = (x, np.min(Ldwn)-1/h, np.max(Ldwn)+1/h)

p = sp.plot(f(x), ran, show=False)

for xi in range(n):
    p.extend(sp.plot(K((x-Ldwn[xi])/h)/(n*h), ran, line_color='r', show=False))

ppl.show(block=False)


from numpy import exp
exec(open("prior2008LDWN.m").read())

f_lbd = sp.lambdify(x, f(x))


def logp_theta(value):
    return np.log(prior2008LDWN(value)).sum()

def logp_est(value, theta):
    return np.log(f_lbd(value)).sum()

with pm.Model() as model:
    theta = pm.DensityDist('theta', logp_theta, testval=np.mean(Ldwnn))
    like = pm.DensityDist('y_est', logp_est, observed={'value': Ldwnn,
                                                       'theta': theta})
    trace = pm.sample(100000)

#pm.traceplot(trace)
#ppl.show(block=False)
pl = sns.distplot(trace.get_values('theta'))
xx = pl.lines[0].get_xdata()
pl.plot(xx, prior2008LDWN(xx))
pl.legend(('posterior', 'posterior', 'prior'))
ppl.show(block=False)
