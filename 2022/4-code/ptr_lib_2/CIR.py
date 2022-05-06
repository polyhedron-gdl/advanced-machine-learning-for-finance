from math import *
import numpy as np


class CIR:

    def __init__(self, **kwargs):
        self.kappa = kwargs["kappa"] 
        self.sigma = kwargs["sigma"] 
        self.theta = kwargs["theta"] 
        self.ro    = kwargs["ro"] 
        self.gamma = sqrt( self.kappa * self.kappa + 2 * self.sigma*self.sigma)
    # --------------

    def B(self, t):
        g = self.gamma
        k = self.kappa
        h = exp(g*t) - 1
        return 2 * h/( (g+k)*h + 2*g)
    # ------------------------

    def A(self, t):
        g = self.gamma
        k = self.kappa
        th= self.theta
        s = self.sigma
        h = exp(g*t) - 1
        return ( 2*k*th/(s*s) ) * log( ( 2 * g * exp(.5 * (k+g)*t) )/( (g+k)*h + 2*g ))
    # --------------------------------------

    def P_tT( self, t, r=None):
        if r == None: r = self.ro
        return exp( -self.B(t)*r + self.A(t) )
        
# ----------------------------------------

def cir_evol( rand, cir, L, dt, Nt, DT, N):

    '''
    @params rand: random number generator
    @params cir :  the CIR object
    @params L   :  number of steps (length) of the cir trajectory simulation
    @params dt  :  step size of the cir trajectory simulation
    @params Nt  :  number of steps of the output trajectory
    @params DT  :  step size for the output trajectory
    @params N   :  number of trajectories
    '''

    zero = np.ndarray(shape = ( L+1 ), dtype=np.double )
    Z    = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    Int  = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    xi   = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))

    s   = cir.sigma
    th  = cir.theta
    k   = cir.kappa
    ro  = cir.ro

    Z[0]   = ro
    Int[0] = 0.0
    zero = 0.0

    for n in range(L):
        Ro = Z[n]
        Rn = Ro + k*(th - Ro)*dt + s*np.sqrt( Ro*dt)*xi[n]
        Io = Int[n]
        Rn = np.maximum(Rn,zero)
        Z[n+1] = Rn
        Int[n+1] = Io + .5*(Rn+Ro)*dt


    #print("dt: %8.4f, Dt: %8.4f" %(dt, DT))
    X  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]
    I  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]

    for n in range(Nt+1):
        tn = n * DT
        pos = int(tn/dt)
        X[n] = Z[pos]
        I[n] = Int[pos]

    return (X, I) 
# ----------------------------------------
