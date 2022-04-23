
import cmath
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

    @property
    def intensity(self): return 1./self.nu
    def compensator(self): return self.phi

    def cf(self, c_k, t):
        # 
        # c_u = i c_k
        #
        c_u = c_k*1j

        c_x = cmath.log( 1.0 -self.nu*self.th*c_u -.5*self.nu*(self.eta*self.eta)*c_u*c_u)
        comp = self.compensator()
        JMP  = t*self.intensity*(comp*c_u - c_x)

        return cmath.exp(JMP)

# -----------------------------------------------------

def vg_evol_step( rand, Sn, vg, Dt, N ):
    nu  = vg.nu
    eta = vg.eta
    th  = vg.th
    I   = vg.intensity
    phi = vg.compensator()
    g   = rand.normal( loc = 0.0, scale = 1.0, size=(N))
    xi  = np.float64( rand.gamma(shape=Dt/nu, scale=nu, size=(N) ))
    X   = th*xi + eta*g*np.sqrt(xi) + Dt*I*phi
    return Sn*np.exp(X)

def vg_evol( rand, So, vg, L, Dt, N ):

    S  = np.ndarray(shape = (L+1, N), dtype=np.double ) # S[N, L] in fortran matrix notation
    S[0] = So
    for n in range(L):
        S[n+1] = vg_evol_step(rand, S[n], vg, Dt, N)

    return S

# ----------------------------------------------------
