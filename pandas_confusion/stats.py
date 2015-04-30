#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from scipy.stats import beta
 
def binom_interval(success, total, confint=0.95):
    """
    Compute two-sided binomial confidence interval in Python. Based on R's binom.test.
    """
    quantile = (1 - confint) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return (lower, upper)

def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def class_agreement(df):
    """
    From e1071 matchClassed.R classAgreement
    """
    n = df.sum().sum()
    nj = df.sum(axis=0)
    ni = df.sum(axis=1)
    m = min(len(ni), len(nj))

    p0 = float(np.diag(df.iloc[0:m, 0:m]).sum()) / n
    pc = float((ni.iloc[0:m] * nj.iloc[0:m]).sum()) / (n**2)

    n2 = choose(n, 2)

    rand = 1 + ((df**2).sum().sum() - ((ni**2).sum() + (nj**2).sum())/2)/n2
    nis2 = ni[ni>1].map(lambda x: choose(int(x), 2)).sum()
    njs2 = nj[nj>1].map(lambda x: choose(int(x), 2)).sum()

    num = df[df>1].dropna(axis=[0,1], thresh=1).applymap(lambda n: choose(n, 2)).sum().sum() - float(nis2 * njs2) / n2
    den = (float(nis2 + njs2) / 2 - float(nis2 * njs2) / n2)
    crand = num / den

    return({
        "diag": p0,
        "kappa": float(p0 - pc) / (1 - pc),
        "rand": rand,
        "crand": crand
    })
    