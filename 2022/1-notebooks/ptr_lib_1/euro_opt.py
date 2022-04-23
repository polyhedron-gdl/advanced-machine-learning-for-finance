#!/usr/bin/env python3

import sys
from sys import stdout as cout
from scipy.stats import norm
from math import *
from .config import get_input_parms

def cn_put( T, sigma, kT):
    if kT == 0.0: return 0.0
    s    = sigma*sqrt(T)
    if s < 1.e-08:
        if kT > 1.0: return 1.
        else       : return 0.
    d    = ( log(kT) + .5*s*s)/s
    return norm.cdf(d)
# ------------------------------------

def an_put( T, sigma, kT):
    if kT == 0.0: return 1.0
    s    = sigma*sqrt(T)
    if s < 1.e-08:
        if kT > 1.0: return 1.0 
        else       : return 0.
    d    = ( log(kT) + .5*s*s)/s
    return norm.cdf(d-s)
# ------------------------------------

'''
    PUT = exp(-rT)Em[ (K - S(T))^+]
    where
    S(T) = So exp( (r-q)*T)*M
    let
    Fw(T) = So exp( (r-q)*T)
    kT    = K/Fw
    then
    PUT = So exp(-qT) Em[ (kT - M)^+]
        = So exp(-qT) FwEuroPut( T, sigma, kT)
'''
def FwEuroPut(T, sigma, kT):
    return ( kT* cn_put( T, sigma, kT) - an_put( T, sigma, kT) )

def FwEuroCall(T, sigma, kT):
    return FwEuroPut(T, sigma, kT) + 1. - kT

def euro_put(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroPut( T, sigma, kT)
# -----------------------

def euro_call(So, r, q, T, sigma, k):
    kT   = exp((q-r)*T)*k/So
    return So*exp(-q*T) * FwEuroCall( T, sigma, kT)
# -----------------------

def recursive_impVol( price, sL, sH, T, kT):
    '''
    pact: price(sL) < price < price(sH)
    '''
    sM = .5*(sH+sL)
    if sH - sL < 1.e-08: return sM
    pM = FwEuroPut(T, sM, kT)
    if fabs( pM - price) < 1.e-10: return sM
    if pM < price: return recursive_impVol( price, sM, sH, T, kT)
    return recursive_impVol( price, sL, sM, T, kT)
# --------------------------------------------


def impVolFromFwPut(price, T, kT):
    if price <= max( kT-1.0,0.0):
        raise Exception("\n\nPrice is too low")
    if price >= 1:
        raise Exception("\n\nPrice is too high")

    sL = 0.0
    pL = FwEuroPut(T, sL, kT)

    sH = 1.0
    while True:
        pH = FwEuroPut(T, sH, kT)
        if fabs( pH - price) < 1.e-10: return sH
        if pH > price: break

    return recursive_impVol( price, sL, sH, T, kT)


def vanilla_options( **keywrds):

    So     = keywrds["S"]
    k      = keywrds["k"]
    r      = keywrds["r"]
    q      = keywrds["q"]
    T      = keywrds["T"]
    sigma  = keywrds["sigma"]
    fp     = keywrds["fp"]

    fp.write("@ %-24s %8.4f\n" %("So", So))
    fp.write("@ %-24s %8.4f\n" %("k", k))
    fp.write("@ %-24s %8.4f\n" %("T", T))
    fp.write("@ %-24s %8.4f\n" %("r", r))
    fp.write("@ %-24s %8.4f\n" %("q", q))
    fp.write("@ %-24s %8.4f\n" %("sigma", sigma))

    kT   = exp((q-r)*T)*k/So
    cnP  = k*exp(-r*T)*cn_put ( T, sigma, kT)
    anP  = So*exp(-q*T)*an_put ( T, sigma, kT)
    put  = euro_put ( So, r, q, T, sigma, k)
    call = euro_call( So, r, q, T, sigma, k)

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
    print("    %-24s: option strike, defaults to .40" %("-v volatility"))
    print("    %-24s: option maturity, defaults to 1.0" %("-T maturity"))
    print("    %-24s: interest rate, defaults to 0.0" %("-r ir"))
# ----------------------------------

def run(args):

    output    = None
    So     = 1.0
    k      = 1.0
    T      = 1.0
    r      = 0.0
    q      = 0.0
    Sigma  =  .2347
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
    except KeyError: pass

    try: k = float( inpts["k"] )
    except KeyError: pass

    try: T = float( inpts["T"] )
    except KeyError: pass

    try: r = float( inpts["r"] )
    except KeyError: pass

    try: Sigma = float( inpts["v"] )
    except KeyError: pass

    res = vanilla_options(fp=fp, T=T, r =r, q=q, sigma=Sigma, k = k, S = So)
    kT = exp((q-r)*T)*k/So
    fwPut = FwEuroPut(T, Sigma, kT)
    impVol = impVolFromFwPut(fwPut, T, kT)

    fp.write("@ Put %14.10f,  Call %14.10f,  Pcn %14.10f,  Pan %14.10f  ImpVol: %10.6f\n" %(res["put"], res["call"], res["cnP"], res["anP"], impVol))

    if output != None: fp.close()
# --------------------------

if __name__ == "__main__":
    run(sys.argv)
