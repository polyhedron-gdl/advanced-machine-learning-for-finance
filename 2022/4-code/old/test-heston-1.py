import numpy as np

from Lib.euro_opt import impVolFromFwPut
from Lib.Heston import Heston
from Lib.FT_opt import ft_opt

def HestonFwPut(Fw, Strike, T, kappa, theta, sigma, ro, rho, Xc = 30):
    
    lmbda = kappa
    nubar = theta
    eta   = sigma  
    nu_o  = ro     # initial value of volatility
    
    hestn = Heston( lmbda = lmbda
                  , eta   = eta
                  , nubar = nubar 
                  , nu_o  = nu_o 
                  , rho   = rho
                  )

    kT  = (Strike/Fw)
    res = ft_opt( hestn, kT, T, Xc)
    
    return res['put'];

def build_smile(strikes=None, Fw=1.0, T= 1.0, Kappa=1., Theta=1., sgma=1.0, Ro=0.0, Rho=0.0, Xc=10):
    vol = {}
    for k in strikes:
        tag = "k=%5.3f" %k
        fwPut = HestonFwPut( Fw = Fw, Strike = k, T = T, kappa=Kappa, theta=Theta, sigma=sgma, ro=Ro, rho=Rho, Xc = Xc)
        if fwPut < max(k-Fw, 0.0): return None
        vol[tag] = impVolFromFwPut(price = fwPut, T = T, kT = k)

    return vol



strikes = np.arange(.4,1.6,.01)
T     = 1.0
theta = 0.04
kappa = 1.5
sigma = 0.3 # eta
rho   = -0.9
ro    = 0.05

feller = sigma*sigma/(2*kappa*theta)

smile = build_smile(strikes, 1.0, T, kappa, theta, sigma, ro, rho, 30)

for k in strikes:
    print(k, HestonFwPut(1.0, k, T, kappa, theta, sigma, ro, rho, Xc = 30),max(k-1, 0.0))
