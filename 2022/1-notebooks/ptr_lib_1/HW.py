#!/usr/bin/env python3

from math import *
from scipy.stats import norm
import numpy as np
# -----------------------------------------------------

class HW:

    def __init__(self, **kwargs):
        self.gamma = kwargs["gamma"] 
        self.sigma = kwargs["sigma"] 
    # --------------------


    def show(self):
        print("@ %-12s: %-8s %8.4f" %("Info", "gamma", self.gamma))
        print("@ %-12s: %-8s %8.4f" %("Info", "sigma", self.sigma))
    # --------------------

    def S2_f(self, t):
        g = self.gamma
        s = self.sigma
        h = exp( -g*t)
        return  ( (s*s)/(g*g ) )* ( t - (2/g)*( 1. - h ) + (1./(2*g)) * (1. - h*h ) ) 
    # ---------------------------

    def S2_X(self, t):
        g = self.gamma
        s = self.sigma
        h = exp( -2*g*t)
        return ( (s*s)/( 2 * g ) )* ( 1. - h )
    # ---------------------------

    def cov(self, t):
        if t < 1./(24*60):
            self.sx  = 0.0
            self.sf  = 0.0
            self.rho = 1.0
            return

        g = self.gamma
        s = self.sigma
        h = exp( -g*t)

        sx  = sqrt( self.S2_X(t) )
        sf  = sqrt( self.S2_f(t) )
        C_xf= ( (s*s)/(2.*g*g) ) * ( 1 - h ) * ( 1 - h )

        self.sx  = sx
        self.sf  = sf
        self.rho = C_xf/(sx*sf)
    # ----------------------------------------

    def BondPrice( self, t, T, dc, x):
        g = self.gamma
        b_tT = (1. - exp(-g*(T-t)))/g
        A_tT = (dc.P_0t(T)/dc.P_0t(t))*exp( .5 * ( self.S2_f(t) + self.S2_f(T-t) - self.S2_f(T) ) )
        return A_tT*np.exp(-b_tT*x)
    # -----------------------------------

    def Annuity( self, t, Dt, p, dc, x):
        A = np.full( len(x), 0.0, dtype=np.double ) 
        for n in range(p):
            A  += Dt*self.BondPrice( t, t+(1+n)*Dt, dc, x)
        return A
    # -----------------------------------

    def SwapRate( self, t, Dt, p, dc, x):
        A = self.Annuity(t, Dt, p, dc, x)
        R = 1. - self.BondPrice( t, t+p*Dt, dc, x)
        return R/A, A
    # -----------------------------------

    def IntPhi( self, dc, Dt, N):
        i_phi  = np.ndarray(shape = N+1, dtype=np.double)

        i_phi[0] = 0
        for n in range(N):
            i_phi[n+1] = i_phi[n] + log( dc.P_0t((n+1)*Dt)/dc.P_0t( n*Dt)) - .5*self.S2_f((n+1)*Dt) + .5 * self.S2_f(n*Dt)

        return i_phi
# ----------------------------------------

    def Sigma( self, t, tm, T):
        '''
        Integrals in the interval [t, tm] of the
        square of the time-dependent volatility of the
        zero coupon bond P(t, T)
        '''
        g = self.gamma
        s = self.sigma
        return ( (s*s)/(2*g*g*g) ) * pow( (exp(-g*T)- exp(-g*tm)), 2) *( exp( 2*g*t) - 1 )
# ----------------------------------------

    def OptionPrice( self, t, T, Strike, dc):
        '''
        Bond put; 
        the option maturity is 't', the bond expires in T
        '''
        g = self.gamma
        s = self.sigma
        S2 = self.Sigma(t, t, T)
        
        S = sqrt( S2 )
        P_ts = dc.P_0t(t)
        P_te = dc.P_0t(T)
        F = Strike*P_ts/P_te
        if S < 1.e-08:
            if F >= 1.:
                P_an = 1.0
                P_cn = 1.0
            else:
                P_an = 0.0
                P_cn = 0.0
        else:
            Lm = log( F )/S - .5*S
            Lp = log( F )/S + .5*S
            P_an = norm.cdf(Lm)
            P_cn = norm.cdf(Lp)

        return P_ts*Strike*P_cn - P_te*P_an
# -----------------------------------
# End Class HW 
# -----------------------------------

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
    This evolution is legal for t <= L Dt.
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
