#!/usr/bin/env python3

from math import *
import numpy as np
from time import time
# -----------------------------------------------------

def mc_heston( rand, So, vol, intVol, cir, rho, Dt, N  ):

    '''
    @parms So    : initial value
    @parms intVol: volatility integral trajectory
    @parms cir   : CIR object
    @parms rho   : correlation between vol and underlying innovations
    @parms Dt    : tenor of the underlying trajectory
                   must agree with the nodes of the volatility trajectory
    @parms N     : number of underlying trajectories
    '''

    # length of the volatility trajectory
    # (including initial point)
    L   = len(intVol)
    th  = cir.theta
    k   = cir.kappa
    eta = cir.sigma
    nu  = vol
    I   = intVol

    # underlying trajectorie
    S  = np.ndarray(shape = (L, N), dtype=np.double ) # S[N, L] in fortran matrix notation

    xi = rand.normal( loc = 0.0, scale = 1.0, size=(L-1, N))

    # prime with So the starting value of each trajectory
    S[0] = So

    for n in range(1,L):
        DI   = I[n] - I[n-1]
        X    = -.5 * DI + (rho/eta)*( nu[n] - nu[n-1] - k*( th*Dt - DI) ) + sqrt((1. - rho*rho)*DI)*xi[n-1]
        S[n] = S[n-1]*np.exp(X)

    return S

# ----------------------------------------------------
