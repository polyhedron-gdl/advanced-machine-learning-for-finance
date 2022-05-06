#!/usr/bin/env python3
import math
import cmath
from math import *
import numpy as np
from Lib.stats import stats

class jumps:

    def __init__(self, **kwargs):
        self._intnsty = kwargs['lmbda']
        self._sgma    = kwargs['sigma']
# -----------------------------------------

    @property
    def intensity(self): return self._intnsty
    @property
    def sigma(self): return self._sgma

    def do_jmp(self, Obj, Dt, J):

        Z   = np.full(shape=J, fill_value=0.0, dtype=np.float)
        Nj  = Obj.poisson(lam=self.intensity * Dt, size=J)
        sup = Nj.max()
        j   = sup

        while j > 0:
            Z = Z + self.single_jump(Obj, Nj >= j)
            j -= 1

        return Z
    # -------------------------------------------------------

    def jd_evol_step(self, rand, Dt, J):

        '''
        Performs 1 step for J trajectories 
        Black-Scholes diffusion + jumps
        '''

        s = self.sigma * sqrt(Dt)
        X = rand.normal( -.5*s*s, s, J)
        X = X + self.do_jmp(rand, Dt, J) + Dt*self.intensity*self.compensator()
        return np.exp(X)

    def cf(self,c_k, t):
        s = self.sigma
        # 
        # c_u = i c_k
        #
        c_u = c_k*1j

        c_x  = -.5 *s*s*c_u*c_u
        comp = -.5 *s*s

        #
        # X_cf = dt * ( u*g - f )  
        #
        X_cf =  t*(comp*c_u - c_x)


        c_x  = self.phi_X(c_u)
        comp = self.compensator()
        JMP  =  t*self.intensity*(comp*c_u - c_x)

        return cmath.exp(X_cf + JMP)

    # =================================================================================


class jmp_binary(jumps):

    '''
    Pr( J==u ) = pi
    Pr( J==d ) = 1. - pi
    '''

    def __init__(self, **kwargs):
        self._pi      = kwargs["pi"] 
        self._u       = kwargs["u"] 
        self._d       = kwargs["d"] 
        super().__init__(**kwargs)
    # -------------------------------------------------------

    def single_jump(self, rand, mask):
            J       = len(mask)
            z       = rand.uniform(low=0.0, high=1.0, size=J)
            pi_mask = np.logical_and(mask, z < self._pi)
            up  = np.where(pi_mask, self._u, 0)

            pi_mask = np.logical_and(mask, z > self._pi)
            down  = np.where(pi_mask, self._d, 0)

            return ( up + down )
    # -------------------------------------------------------

    def compensator(self):
        phi_J =  self._pi*exp(self._u) + (1.0 - self._pi)*exp(self._d) 
        return (1.0 - phi_J)

    def phi_X(self, c_u):
        return 1. - self._pi*cmath.exp(self._u*c_u) - (1.-self._pi)*cmath.exp(self._d*c_u)
        
# ================================================================================

class jmp_normal(jumps):

    '''
    P( J < L ) = N_{0,1}( (L - m)/eta )
    '''

    def __init__(self, **kwargs):
        self._m       = kwargs["m"] 
        self._eta     = kwargs["eta"] 
        super().__init__(**kwargs)

    def single_jump(self, rand, mask):
            J       = len(mask)
            X   = rand.normal( self._m, self._eta, J)
            return np.where(mask, X, 0.)
    # -------------------------------------------------------

    def compensator(self):
       phi_J = exp( self._m + .5*(self._eta)*(self._eta)); 
       return (1.0 - phi_J)

    def phi_X(self, c_u):
        c_z = self._m*c_u + .5 * self._eta*self._eta*c_u*c_u
        return 1. - cmath.exp(c_z)

