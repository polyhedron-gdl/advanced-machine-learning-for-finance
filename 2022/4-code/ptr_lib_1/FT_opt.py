import math
from math import *

def pr_x_lt_w ( self, Xc, w, off, t):

    m = 1
    tot = 0.0
    while True:
        c_k = 2*math.pi*( m/(2*Xc) + off )
        c_phi = self.cf(c_k, t)
        th    = math.pi * m * w/Xc;
        delta = (cos(th)*c_phi.imag - sin(th)*c_phi.real)/m; 
        tot  += delta
        if fabs(delta/tot) < 1.e-10: break
        m += 2
    return .5 - 2.*tot/math.pi

def ft_opt(self, Strike, T, Xc):
    w       = log(Strike)

    off = complex(0.0, 0.0)
    cn = pr_x_lt_w( self, Xc, w, off, T)

    off = complex(0.0, -1/(2*math.pi))
    an = pr_x_lt_w( self, Xc, w, off, T)

    
    put  = Strike*cn - an; 
    pcn  = cn;
    pan  = an;
    call = put + (1. - Strike);

    return {"put": put, "call":  call, "pCn": pcn, "pAn": pan}
