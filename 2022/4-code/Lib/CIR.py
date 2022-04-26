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

        #
        # when g >> 1 we do neglect terms of the 
        # type g*exp(-gt)
        # the situation g >> 1 occurs only when we try to test 
        # very large violation from the Feller condition
        #
        if g > 30: return 2 /(g+k)
        h = exp(g*t) - 1
        return 2 * h/( (g+k)*h + 2*g)
    # ------------------------

    def A(self, t):
        g  = self.gamma
        k  = self.kappa
        th = self.theta
        s  = self.sigma
        #
        # when g >> 1 we do neglect terms of the 
        # type g*exp(-gt)
        # the situation g >> 1 occurs only when we try to test 
        # very large violation from the Feller condition
        #
        if g > 30:
            return ( 2*k*th/(s*s) ) * ( log( 2 * g ) + .5 * (k+g)*t - g*t + log(g+k))

        h = exp(g*t) - 1
        return ( 2*k*th/(s*s) ) * ( log( 2 * g ) + .5 * (k+g)*t - log( (g+k)*h + 2*g) )
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
# ----------------------------------------------------------


##
## From: "Efficient Simulation of the Heston Stochastic Volatility Model" by Leif Andersen
## the algorithm here implementd is denoted as 'QE' in the paper
##
def QT_cir_evol( rand, cir, L, dt, Nt, DT, N):

    s   = cir.sigma
    th  = cir.theta
    k   = cir.kappa
    ro  = cir.ro

    PSI_c    = 1.5
    V      = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    In     = np.ndarray(shape = ( L+1, N ), dtype=np.double )
    xi     = rand.normal( loc = 0.0, scale = 1.0, size=(L, N))
    V[0]   = ro
    In[0]  = 0.0

    for n in range(L):
        Zero = V[n] == 0.0
        V[n+1]= np.where(Zero,k*th*dt, 0.0)

        h   = 1. - exp(-k*dt)
        m   = th + ( V[n] - th)*(1. - h)
        s2  = (s*s*h/k)*( V[n] * (1. - h ) + .5*th*h )
        PSI = s2/(m*m)

        #V[n+1] = 0.0
        Mask   = np.logical_and( PSI > PSI_c, ~Zero )
        u      = rand.uniform(low=0.0, high=1.0, size = N)
        p      = (PSI-1)/(PSI+1)
        opMask = np.logical_and( u > p, Mask == 1 )
        beta   = (1. - p)/m
        x      = np.where(opMask, np.log( (1-p)/(1-u))/beta, 0.0)

        Mask   = np.logical_and( PSI <= PSI_c, ~Zero )
        o      = np.where( Mask, 2/PSI - 1., 0.0)
        b2     = np.where(Mask, o + np.sqrt(o*(o+1)), 0.0) 
        a      = m/(1. + b2)
        c      = np.power( (np.sqrt(b2)+ xi[n]), 2, where=Mask)
        y      = np.where(Mask, a*c, 0)


        V[n+1] += (x + y)
        In[n+1] = In[n] + (dt/2.) * ( V[n]  + V[n+1] )

    X  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]
    I  = np.ndarray(shape = ( Nt+1, N ), dtype=np.double ) # Y[Nt+1, 2]

    for n in range(Nt+1):
        tn = n * DT
        pos = int(tn/dt)
        X[n] = V[pos]
        I[n] = In[pos]

    return  (X,I)
# ----------------------------------------
