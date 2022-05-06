
from math import *
import numpy as np


class VG:

    def __init__(self, **kwargs):
        nu  = kwargs["nu"]
        eta = kwargs["eta"]
        th  = kwargs["theta"]

        self.phi = log( 1. - nu*th - .5*nu*eta*eta)
        self.nu  = nu
        self.eta = eta
        self.th  = th
    #-----------------

    def intensity(self): return 1./self.nu
    def compensator(self): return self.phi

# -----------------------------------------------------

def vg_evol( rand, So, vg, L, Dt, N ):

    nu  = vg.nu
    eta = vg.eta
    th  = vg.th
    I   = vg.intensity()
    phi = vg.compensator()

    # underlying trajectories
    S  = np.ndarray(shape = (L+1, N), dtype=np.double ) # S[N, L] in fortran matrix notation
    g  = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    xi = np.float64( rand.gamma(shape=Dt/nu, scale=nu, size=(L,N) ))
    

    # prime with So the starting value of each trajectory
    S[0] = So

    for n in range(L):
        X      = th*xi[n] + eta*g[n]*np.sqrt(xi[n]) + Dt*I*phi
        S[n+1] = S[n]*np.exp(X)

    return S

# ----------------------------------------------------
