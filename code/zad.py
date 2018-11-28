import numpy as np
from numpy import exp
import sympy as sp
from sympy.abc import i,x
import pymc3 as pm
import seaborn as sns
import matplotlib as mpl
#mpl.use("pgf")
import matplotlib.pyplot as ppl
import platform, locale

if platform.system() == 'Windows':
        locale.setlocale(locale.LC_NUMERIC, 'Polish')
else:
        locale.setlocale(locale.LC_NUMERIC, ('pl_PL', 'UTF-8'))

r29 = np.loadtxt("rok2009.txt")
exec(open("prior2008LDWN.py").read())
exec(open("prior2008LN.py").read())

n = 10

Ldwnn = np.random.choice(r29[:, 0], n)
Lnn = np.random.choice(r29[:, 1], n)

Ldwn = sp.Array(Ldwnn)
Ln = sp.Array(Lnn)


def f(x, xo, K=None, h=None):
    n = len(xo)
    if h is None:
        h = 1.06*np.array(xo, dtype=float).std()*n**(-1/5)
    if K is None:
        K = lambda x: sp.exp(-(x**2)/2)/sp.sqrt(2*sp.pi)
    return 1/(n*h)*sp.Sum(K((x - xo[i-1])/h), (i, 1, n)).doit()


def kernelPlot(xo, K=None, h=None):
    if h is None:
        h = 1.06*np.array(xo, dtype=float).std()*n**(-1/5)

    if K is None:
        K = lambda x: sp.exp(-(x**2)/2)/sp.sqrt(2*sp.pi)

    ran = (x, np.min(xo)-1/h, np.max(xo)+1/h)

    p = sp.plot(f(x, xo), ran, show=False)

    for xi in range(len(xo)):
        p.extend(sp.plot(K((x-xo[xi])/h)/(h*len(xo)), ran, line_color='r', show=False))

    return p


def bayesEst(xo, prior):

    f_lbd = sp.lambdify(x, f(x, xo))

    def logp_theta(value):
        return np.log(prior(value)).sum()

    def logp_est(value):
        return np.log(f_lbd(value)).sum()

    with pm.Model() as model:
        theta = pm.DensityDist('theta', logp_theta, testval=np.mean(xo))
        kern = pm.DensityDist('kern', logp_est, observed={'value': np.array(xo, dtype=float)})
        trace = pm.sample(25000, nuts_kwargs={'target_accept': 0.9})

    return trace




#pl = sns.distplot(trace.get_values('theta'))
#xx = pl.lines[0].get_xdata()
#pl.plot(xx, prior2008LDWN(xx))
#pl.legend(('posterior', 'posterior', 'prior'))
#ppl.show(block=False)
