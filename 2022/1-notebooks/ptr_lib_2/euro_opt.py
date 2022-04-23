#!/usr/bin/env python3

import sys
from sys import stdout as cout
from scipy.stats import norm
import numpy as np
from math import *
from .config import get_input_parms

#
# NumPy enabled asset or nothing
#
def __npan__ ( Fw, T, sgma, k, Shft=0.0):
    s          = sgma*np.sqrt(T)
    mask       = np.where( s < 1.e-08, 1, 0)
    MASK       = np.where( s >= 1.e-08, 1, 0)
    s          = np.where(mask, 1., s)
    dm         = ( np.log( (k+Shft)/(Fw+Shft)) -.5*s*s)/s
    an         = norm.cdf(dm)

    m1         = np.logical_and(mask, Fw <= k)
    res        = np.where(m1, 1., 0.)
    res       += np.where(MASK,an,0.0);

    return res
# ------------------------------------

#
# NumPy enabled cash or nothing
#
def __npcn__ ( Fw, T, sgma, k, Shft=0.0):
    s          = sgma*np.sqrt(T)
    #print("npcn: s = %f" %s)
    mask       = np.where( s < 1.e-08, 1, 0)
    MASK       = np.where( s >= 1.e-08, 1, 0)
    s          = np.where(mask, 1., s)
    dp         = ( np.log( (k+Shft)/(Fw+Shft)) + .5*s*s)/s
    cn         = norm.cdf(dp)

    m1         = np.logical_and(mask, Fw <= k)
    res        = np.where(m1, 1., 0.)
    res       += np.where( MASK,cn,0.0);

    return res
# ------------------------------------

def np_fw_euro_put(F, T, sgma, k, Shft=0):
    return (k+Shft)*__npcn__( F, T, sgma, k, Shft=Shft)  - (F+Shft)*__npan__( F, T, sgma, k, Shft=Shft) 
# ---------------------------

def np_fw_euro_call(F, T, sgma, k, Shft=0):
    return np_fw_euro_put(F, T, sgma, k, Shft=Shft) + F - k
# ---------------------------

def np_euro_put(So, r, T, sigma, k, Shft=0.0):
    Fw = np.exp(r*T)*So
    return np.exp(-r*T)*np_fw_euro_put(Fw, T, sigma, k, Shft=Shft)
# -----------------------

def np_euro_call(So, r, T, sigma, k, Shft=0.0):
    Fw = np.exp(r*T)*So
    return np.exp(-r*T)*np_fw_euro_call(Fw, T, sigma, k, Shft=Shft)
# -----------------------

    

def __an__ ( Fw, T, sgma, k, Shft=0.0):
    s = sgma*sqrt(T)
    if s < 1.e-08:
        if Fw <= k: return 1
        else:      return 0.0

    dm = ( log( (k+Shft)/(Fw+Shft)) -.5*s*s)/s
    return norm.cdf(dm)
# ------------------------------------

def __cn__ ( Fw, T, sgma, k, Shft=0.0):
    s = sgma*sqrt(T)
    if s < 1.e-08:
        if Fw <= k: return 1
        else:      return 0.0

    dp = ( log( (k+Shft)/(Fw+Shft)) +.5*s*s)/s
    return norm.cdf(dp)
# ------------------------------------

def fw_euro_put(F, T, sgma, k, Shft=0):
    return (k+Shft)*__cn__( F, T, sgma, k, Shft=Shft)  - (F+Shft)*__an__( F, T, sgma, k, Shft=Shft) 
# ---------------------------

def fw_euro_call(F, T, sgma, k, Shft=0):
    return fw_euro_put(F, T, sgma, k, Shft=Shft) + F - k
# ---------------------------

def cn_put(So, r, T, sigma, k, Shft=0.0):
    Fw = exp(r*T)*So
    return exp(-r*T)*__cn__ ( Fw, T, sigma, k, Shft=Shft)
# ------------------------------------

def an_put(So, r, T, sigma, k, Shft=0.0):
    Fw = exp(r*T)*So
    return exp(-r*T)*(-Shft*__cn__(Fw,T,sigma,k,Shft=Shft) + (Fw+Shft)*__an__(Fw, T, sigma, k, Shft=Shft) )
# ------------------------------------

def euro_put(So, r, T, sigma, k, Shft=0.0):
    Fw = exp(r*T)*So
    return exp(-r*T)*fw_euro_put(Fw, T, sigma, k, Shft=Shft)
# -----------------------

def euro_call(So, r, T, sigma, k, Shft=0.0):
    put = euro_put( So, r, T, sigma, k, Shft=Shft)
    return put + So - exp(-r*T)*k
# -----------------------

def vanilla_options( **keywrds):

    So     = keywrds["S"]
    k      = keywrds["k"]
    r      = keywrds["r"]
    T      = keywrds["T"]
    sigma  = keywrds["sigma"]
    fp     = keywrds["fp"]
    try: Shft = keywrds["Shft"]
    except KeyError: Shft = 0.0

    fp.write("@ %-24s %8.4f\n" %("So", So))
    fp.write("@ %-24s %8.4f\n" %("k", k))
    fp.write("@ %-24s %8.4f\n" %("T", T))
    fp.write("@ %-24s %8.4f\n" %("r", r))
    fp.write("@ %-24s %8.4f\n" %("sigma", sigma))

    cnP  = cn_put ( So, r, T, sigma, k, Shft=Shft)
    anP  = an_put ( So, r, T, sigma, k, Shft=Shft)
    put  = euro_put ( So, r, T, sigma, k, Shft=Shft)
    call = euro_call( So, r, T, sigma, k, Shft=Shft)

    return {"put": put, "call": call, "anP": anP, "cnP": cnP}
# --------------------------
def usage():
    print("Computes the value of european Call/Put options")
    print("and put-cash or nothing and put asset or nothing")
    print("Usage: $> ./euro_opt.py [options]")
    print("Options:")
    print("    %-24s: this output" %("--help"))
    print("    %-24s: output file" %("-out outputFile"))
    print("    %-24s: initial value of the underlying, defaults to 1.0" %("-s So"))
    print("    %-24s: option strike, defaults to 1.0" %("-k strike"))
    print("    %-24s: shift for log-normal shifted model, defaults to 0.0" %("-shft Shift"))
    print("    %-24s: option strike, defaults to .40" %("-v volatility"))
    print("    %-24s: option maturity, defaults to 1.0" %("-T maturity"))
    print("    %-24s: interest rate, defaults to 0.0" %("-r ir"))
# ----------------------------------

def run(args):

    output    = None
    So     = 1.0
    T      = 1.0
    r      = 0.0
    Sigma  =  .40
    inpts  = get_input_parms(args)

    try:
        Op = inpts["help"]
        usage()
        return
    except KeyError:
        pass

    try:
        output = inpts["out"]
        fp     = open(output, "w")
    except KeyError:
        fp     = cout

    try: So = float( inpts["So"] )
    except KeyError: So = 1.0

    try: k = float( inpts["k"] )
    except KeyError: k=1.0

    try: T = float( inpts["T"] )
    except KeyError: pass

    try: r = float( inpts["r"] )
    except KeyError: pass

    try: Sigma = float( inpts["v"] )
    except KeyError: pass

    try: Shft = float( inpts["shft"] )
    except KeyError: Shft = 0.0

    res = vanilla_options(fp=fp, T=T, r =r, sigma=Sigma, k = k, S = So, Shft=.3)
    fp.write("@ Put %14.10f,  Call %14.10f,  Pcn %14.10f,  Pan %14.10f\n" %(res["put"], res["call"], res["cnP"], res["anP"]))

    if output != None: fp.close()
# --------------------------

if __name__ == "__main__":
    run(sys.argv)
