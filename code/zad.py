import numpy as np
from numpy import exp
import sympy as sp
from sympy.abc import i,x
import pymc3 as pm
import seaborn as sns
import matplotlib as mpl
mpl.use("pgf")
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

    f_lbd = sp.lambdify(x, f(x, xo))
    K_lbd = sp.lambdify(x, K(x))

    ran = np.arange(np.min(xo)-1/h, np.max(xo)+1/h, 0.01, dtype=float)

    ppl.figure()
    ppl.plot(ran, f_lbd(ran))

    for xi in range(len(xo)):
        ppl.plot(ran, K_lbd((ran-float(xo[xi]))/h)/(h*len(xo)), '--')


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


def classEst(data, k = 1.96):

    Ei = np.power(10,0.1*data)

    E_m = np.mean(Ei)

    L_m = 10*np.log10(E_m)

    n = np.sum(data)

    uE = np.std(Ei)/np.sqrt(n)

    Lu = 10*np.log10(E_m + k*uE)
    Ll = 10*np.log10(E_m - k*uE)

    U95u = Lu - L_m
    U95l = L_m - Ll
    return U95l, L_m,  U95u


Ldwn_tr = bayesEst(Ldwn, prior2008LDWN)
Ln_tr = bayesEst(Ln, prior2008LN)
#1. Tabela

#2. Wykresy

ppl.figure()
ppl.hist(Ldwn_tr.get_values('theta'), bins=30)
ppl.xlabel(r"Poziom ciśnienia [\si{\decibel}]")
ppl.ylabel(r"Liczność")
ppl.savefig("../report/plots/hist_Ldwn.pgf")

ppl.figure()
ppl.hist(Ln_tr.get_values('theta'), bins=30)
ppl.xlabel(r"Poziom ciśnienia [\si{\decibel}]")
ppl.ylabel(r"Liczność")
ppl.savefig("../report/plots/hist_Ln.pgf")

kernelPlot(Ldwn)
ppl.xlabel(r"Poziom ciśnienia [\si{\decibel}]")
ppl.ylabel(r"Prawdopodobieństwo")
ppl.savefig("../report/plots/kernel_Ldwn.pgf")

kernelPlot(Ln)
ppl.xlabel(r"Poziom ciśnienia [\si{\decibel}]")
ppl.ylabel(r"Prawdopodobieństwo")
ppl.savefig("../report/plots/kernel_Ln.pgf")

#pl = sns.distplot(trace.get_values('theta'))
#xx = pl.lines[0].get_xdata()
#pl.plot(xx, prior2008LDWN(xx))
#pl.legend(('posterior', 'posterior', 'prior'))
#ppl.show(block=False)
