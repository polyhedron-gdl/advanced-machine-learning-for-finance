#!/usr/bin/env python3

from math import *
import numpy as np
# -----------------------------------------------------

def hw_evol(rand, hw, Dt, L, N):

    '''
    Evolution of the H+W model in the 'bank-account' numeraire
    '''

    hw.cov(Dt)

    xr = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    ir = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))

    X  = np.ndarray(shape = ( L+1, N), dtype=np.double )
    Ix = np.ndarray(shape = ( L+1, N), dtype=np.double )

    sx   = hw.sx
    sf   = hw.sf
    rho  = hw.rho
    gamm = hw.gamma
    g    = 1 - exp(-gamm*Dt)

    ir = sf*( rho*xr + sqrt( 1. - rho*rho)*ir )
    xr = sx*xr

    X[0]  = 0.0
    Ix[0] = 0.0

    for n in range(L):
        mx      = - g*(X[n])
        mf      =  (g/gamm)*(X[n])
        X[n+1]  = X[n]  + mx + xr[n]
        Ix[n+1] = Ix[n] + mf + ir[n]

    return X, Ix
# ----------------------------------------

def hw_evol_P_0T(rand, hw, Dt, L, T, N):

    '''
    Evolution of the H+W model in the 'terminal P_(t,T)' numeraire
    '''

    xr = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    X  = np.ndarray(shape = ( L+1, N), dtype=np.double )

    sx   = sqrt(hw.S2_X(Dt))
    gamm = hw.gamma
    sgma = hw.sigma
    g    = exp(-gamm*Dt)
    h    = (1. - exp(-gamm*Dt))/gamm
    h2   = (1. - exp(-2*gamm*Dt))/(2*gamm)
    xr   = sx*xr

    X[0]  = 0.0

    for n in range(L):
        mx      = g*X[n] - ((sgma*sgma)/gamm)*(h - exp(-gamm*(T-(n+1)*Dt))*h2)
        X[n+1]  = mx + xr[n]

    return X
# ----------------------------------------
