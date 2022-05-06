from math import *
from Lib.euro_opt import cn_put, an_put

def cn_put_delta( S, r, T, sigma, B, M):
    mu = r - .5*sigma*sigma
    g  = 2.*mu/(sigma*sigma)
    return cn_put( S, r, T, sigma, M) - exp( g * log(B/S)) *cn_put( B*B/S, r, T, sigma, M)
# --


def an_put_delta( S, r, T, sigma, B, M):
    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    return an_put( S, r, T, sigma, M) - exp( g * log(B/S)) *an_put( B*B/S, r, T, sigma, M);
# --

def cn_put_ko( S, r, T, sigma, k, B):

    if S < B :     # High barrier 
        if k < B: M = k
        else:     M = B
        return cn_put_delta( S, r, T, sigma, B,  M);

    if k < B :  
        return 0.0;

    return cn_put_delta( S, r, T, sigma, B,  k) - cn_put_delta( S, r, T, sigma, B,  B);
# ---

def cn_call_ko( S, r, T, sigma, k, B):

    if S < B: 
        return cn_put_delta( S, r, T, sigma, B, B) - cn_put_ko( S, r, T, sigma, k,  B);

    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    f  = exp( g * log(B/S)) ;
    return ( 1.0 - f )  - cn_put_delta( S, r, T, sigma, B, B) - cn_put_ko( S, r, T, sigma, k,  B);
# ---

def an_put_ko( S, r, T, sigma, k, B):

    if S < B:      # High barrier 
        if k < B: M = k
        else:     M = B
        return an_put_delta( S, r, T, sigma, B,  M);

    if k < B: return 0.0;
    return an_put_delta( S, r, T, sigma, B,  k) - an_put_delta( S, r, T, sigma, B,  B);
# --

def an_call_ko( S, r, T, sigma, k, B):

    if S < B: 
        return an_put_delta( S, r, T, sigma, B, B) - an_put_ko( S, r, T, sigma, k,  B);

    mu = r - .5*sigma*sigma;
    g  = 2.*mu/(sigma*sigma);
    f  = exp( g * log(B/S)) ;
    return ( S - (B*B/S)*f ) - an_put_delta( S, r, T, sigma, B, B) - an_put_ko( S, r, T, sigma, k,  B);
# -----

# Knock out put option
def put_ko( S, r, T, sigma, k, B):
    return k * cn_put_ko( S, r, T, sigma, k, B) - an_put_ko( S, r, T, sigma, k, B);

# Knock out call option
def call_ko( S, r, T, sigma, k, B):
    return an_call_ko( S, r, T, sigma, k, B) - k * cn_call_ko( S, r, T, sigma, k, B)
