import numpy  as np
import pandas as pd
import time
import math

from sklearn.model_selection import train_test_split

from ptr_lib_1.Heston import Heston
from ptr_lib_1.FT_opt import ft_opt
from Heston_utils     import lhs_sampling, histo_params

from tqdm             import tqdm
#__________________________________________________________________________________________
#
def HestonPut(St, Strike, T, kappa, theta, sigma, v0, rho, r, Xc = 30):

    kT    = (Strike/St)*math.exp(-r*T)
    
    hestn = Heston(lmbda=kappa, eta=sigma, nubar=theta, nu_o=v0, rho=rho)
    res   = ft_opt(hestn, kT, T, Xc)
    
    return res['put'];
#__________________________________________________________________________________________
#
def parms_gen( lhs = None, Xc=10, strikes=None):

    if lhs is None: raise Exception("No data to process")
    x = lhs

    NUM = len(x["T"])

    X = pd.DataFrame()
    for tag in list(x):
        X[tag] = np.full(NUM,0.0, dtype = np.double)
    X["Price"] = np.full(NUM,0.0, dtype = np.double)
    
    __tStart = time.perf_counter()
    pCount = 0
    cCount = 0
    n      = 0
    
    for m in tqdm(range(NUM)):
        Fw    = 1.0
        K     = x["Strike"][m]

        fwPut = HestonPut( St     = Fw
                         , Strike = K
                         , T      = x["T"][m]
                         , kappa  = x["k"][m]
                         , theta  = x["theta"][m]
                         , sigma  = x["sigma"][m]
                         , v0     = x["v0"][m]
                         , r      = 0 
                         , rho    = x["rho"][m]
                         , Xc     = Xc)
        
        if fwPut < max(K-Fw,0.): 
            pCount += 1
            continue

        for tag in list(x):
            X[tag][n] = x[tag][m]
        X["Price"][n] = fwPut
        n += 1
        # ---------------------------------------

    __tEnd = time.perf_counter()
    print("@ %-34s: elapsed %.4f sec" %("Seq. pricing", __tEnd - __tStart) )

    # Trim the original vector ....
    nSamples = n

    df = pd.DataFrame()
    for s in X.keys(): df[s] = np.copy(X[s][0:nSamples])
    print("@ %-34s: Violations Put=%d, Call=%d DB=%d out of %d" %("Info", pCount, cCount, nSamples, NUM))
    return df
#__________________________________________________________________________________________
#
verbose = False
    
outputPrfx    = "full"
testPrfx      = "test"
targetPrfx    = "trgt"
    
EPS           = 0.00
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

NUM     = 1024
TAG     = str(NUM) + '_MCA'
rand    = np.random.RandomState(42)

# strikes used to build the smile used as a regressor
strikes = np.arange(.8, 1.2, .025)

__tStart = time.perf_counter()
xDF = lhs_sampling(rand, NUM, bounds = bounds)
__tEnd = time.perf_counter()
print("@ %-34s: elapsed %.4f sec" %("LHS", __tEnd - __tStart) )

# Let's check the distribution of the parameters we have generated
histo_params( xDF, title = "Random parameters density functions")

# Generate training/test set
__tStart = time.perf_counter()
df =  parms_gen( lhs = xDF, Xc=XC, strikes = strikes)
__tEnd = time.perf_counter()
print("@ %-34s: elapsed %.4f sec" %("GEN", __tEnd - __tStart) )

X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)

Y_test = pd.DataFrame({"Price": X_test["Price"]})
X_test = X_test.drop(columns="Price")

outputFile = "%s_%s.csv" %(outputPrfx, TAG)
X_train.to_csv(outputFile, sep=',', float_format="%.6f", index=False)
print("@ %-34s: training data frame written to '%s'" %("Info", outputFile))

challengeFile = "%s_%s.csv" %(testPrfx, TAG)
X_test.to_csv(challengeFile, sep=',', float_format="%.6f", index=False)
print("@ %-34s: challenge data frame written to '%s'" %("Info", challengeFile))

targetFile = "%s_%s.csv" %(targetPrfx, TAG)
Y_test.to_csv(targetFile, sep=',', float_format="%.6f", index=False)
print("@ %-34s: target data frame written to '%s'" %("Info", targetFile))
    
