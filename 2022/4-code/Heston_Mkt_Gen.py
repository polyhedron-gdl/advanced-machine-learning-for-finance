import numpy  as np
import pandas as pd
import time
import math

from sklearn.model_selection import train_test_split

from ptr_lib_1.Heston   import Heston
from ptr_lib_1.FT_opt   import ft_opt
from ptr_lib_1.euro_opt import impVolFromFwPut
from Heston_utils       import lhs_sampling_2, histo_array

from tqdm               import tqdm
#__________________________________________________________________________________________
#
def HestonPut(St, Strike, T, kappa, theta, sigma, v0, rho, r, Xc = 30):

    kT    = (Strike/St)*math.exp(-r*T)
    
    hestn = Heston(lmbda=kappa, eta=sigma, nubar=theta, nu_o=v0, rho=rho)
    res   = ft_opt(hestn, kT, T, Xc)
    
    return res['put'];
#__________________________________________________________________________________________
#
def build_smile(strikes=None, Fw=1.0, T= 1.0, Kappa=1., Theta=1., sgma=1.0, Ro=0.0, Rho=0.0, Xc=10):
    vol = {}
    for k in strikes:
        tag = "k=%5.3f" %k
        fwPut = HestonPut(Fw, k, T, Kappa, Theta, sgma, Ro, Rho, 0.0, Xc)
        if fwPut < max( k-Fw, 0.0): return None
        vol[tag] = impVolFromFwPut(price = fwPut, T = T, kT = k)

    return vol
#__________________________________________________________________________________________
#
def mkt_gen(pars = None, kw = None, Xc=10, strikes=None):

    if pars is None: raise Exception("No data to process")
    if kw   is None: raise Exception("No list of tags")
    x = pars

    NUM = len(x[:,0])
    
    X = {}
    for k in strikes:
        tag    = "k=%5.3f" %k
        X[tag] = np.full(NUM,0.0, dtype = np.double)
    
    X["T"]      = np.full(NUM,0.0, dtype = np.double)
    X["Price"]  = np.full(NUM,0.0, dtype = np.double)
    X["Strike"] = np.full(NUM,0.0, dtype = np.double)
    
    __tStart=time.perf_counter()
    pCount = 0
    cCount = 0
    n      = 0
    
    for m in tqdm(range(NUM)):
        Fw    = 1.0
        Kappa = x[m,0]
        Theta = x[m,1]
        sgma  = x[m,2]
        Ro    = x[m,3]
        Rho   = x[m,4]
        # --
        T     = x[m,5]
        K     = x[m,6]

        fwPut = HestonPut(Fw, K, T, Kappa, Theta, sgma, Ro, Rho, 0.0, Xc = Xc)
        if fwPut < max(K-Fw,0.): 
            pCount += 1
            continue

        vol = build_smile(strikes=strikes, Fw=Fw, T= T, Kappa=Kappa, Theta=Theta, sgma=sgma, Ro=Ro, Rho=Rho, Xc=Xc)
        if vol == None: 
            cCount += 1
            continue

        for k in strikes:
            tag       = "k=%5.3f" %k
            X[tag][n] = vol[tag]
        
        X["Price"][n]  = fwPut
        X["Strike"][n] = K
        X["T"][n]      = T
        n += 1
        # ---------------------------------------

    __tEnd = time.perf_counter()
    print("@ %-34s: elapsed %.4f sec" %("Seq. pricing", __tEnd - __tStart) )

    nSamples = n

    df = {}
    for s in X.keys():
        df[s] = np.copy(X[s][0:nSamples])
    print("@ %-34s: Violations Put=%d, Call=%d DB=%d out of %d" %("Info", pCount, cCount, nSamples, NUM))
    return pd.DataFrame(df)
#__________________________________________________________________________________________
#
verbose = False
    
outputPrfx    = "full"
challengePrfx = "test"
targetPrfx    = "trgt"
    
EPS           = 0.01
XC            = 10.0
    
# bounds for the random generation of model parameters
# and contract parameters
bounds = { "k":       [ .01   , 1.00]
         , "theta":   [ .01   ,  .80]
         , "sigma":   [ .01   , 1.00]
         , "v0":      [ .01   ,  .80]
         , "rho":     [-.99   , 0.00]
         , "T":       [ 1./12., 2.00]
         , "Strike":  [ .6    , 1.40]
         }

    
NUM     = 4096
TAG     = str(NUM) + "_VFA"
rand    = np.random.RandomState(42)

__tStart= time.perf_counter()
kw, x = lhs_sampling_2(rand, NUM, bounds = bounds)
__tEnd = time.perf_counter()
print("@ %-34s: elapsed %.4f sec" %("LHS", __tEnd - __tStart) )
    
# Let's check the distribution of the parameters we have generated
histo_array(list(kw), x, title = "Random parameters density functions") 

# strikes used to build the smile used as a feature (regressor)
strikes = np.arange(.8, 1.2, .025)
    
# Generate training/test set
__tStart=time.perf_counter()
df =  mkt_gen(pars = x, kw = kw, Xc=XC, strikes = strikes)
__tEnd = time.perf_counter()
print("@ %-34s: elapsed %.4f sec" %("GEN", __tEnd-__tStart) )

X, Y = train_test_split(df, test_size=0.33, random_state=rand.randint(42))

t = pd.DataFrame({"Price": Y["Price"]})
y = Y.drop(columns="Price") 

outputFile = "%s_%s.csv" %(outputPrfx, TAG)
X.to_csv(outputFile, sep=',', float_format="%.6f", index=False)
print("@ %-34s: training data frame written to '%s'" %("Info", outputFile))

challengeFile = "%s_%s.csv" %(challengePrfx, TAG)
y.to_csv(challengeFile, sep=',', float_format="%.6f", index=False)
print("@ %-34s: challenge data frame written to '%s'" %("Info", challengeFile))

targetFile = "%s_%s.csv" %(targetPrfx, TAG)
t.to_csv(targetFile, sep=',', float_format="%.6f", index=False)
print("@ %-34s: target data frame written to '%s'" %("Info", targetFile))
