import pandas as pd
import numpy  as np
import scipy

from smt.sampling_methods import LHS

def lhs_sampling(rand, NUM, bounds=None):

    mInt = 2**15
    MInt = 2**16
    kw   = list(bounds)

    # builds the array of bounds
    limits = np.empty(shape=(0,2))
    for k in kw: 
        limits = np.concatenate((limits, [bounds[k]]), axis=0)

    sampling = LHS(xlimits=limits, random_state=rand.randint(mInt,MInt))
    x        = sampling(NUM)

    X = pd.DataFrame()
    for n in range(len(kw)):
        tag    = kw[n]
        X[tag] = x[:,n]

    return X
    
bounds = {  "T"     : [1./12., 2.00]
          , "Sigma" : [ .01  ,  .80]
          , "Strike": [ .4   , 1.20]
         }

# Number of Observations
NUM = 100000
# Random number generator
rand = np.random.RandomState(42)
# Latin Hypercube Sampling
xDF = lhs_sampling(rand, NUM, bounds=bounds)
    