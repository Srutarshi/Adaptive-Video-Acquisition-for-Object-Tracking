import numpy as np
import math as ma
import argparse
import time
import copy

def qt_sets(Nl):
    Ns = Nl
    
    f = [None for i in range(Ns)]
    t = [None for i in range(Ns)]
    for n in range(1,Ns+1):
        f[n-1] = [None for i in range(4**(Ns-n))]
        t[n-1] = [None for i in range(4**(Ns-n))]

    f[Ns-1][0] = np.array([[Ns, 1]])
    t[Ns-1][0] = np.array([[Ns, 1]])
    #f,t = sets(Ns-1,0,f,t)
    f,t = sets(Ns,0,f,t)
    return f,t


def sets(l,i,f,t):
    f[l-2][4*i] = np.array([[l-1, (4*i)+1]])
    f[l-2][(4*i)+1] = np.array([[l-1, (4*i)+2]])
    f[l-2][(4*i)+2] = np.array([[l-1, (4*i)+3]])
    f[l-2][(4*i)+3] = np.vstack((f[l-1][i], np.array([[l-1,4*i+4]])))     #this is line #22 of MATLAB code
    
    t[l-2][(4*i)] = np.vstack((t[l-1][i], np.array([[l-1, (4*i)+1]])))
    t[l-2][(4*i)+1] = np.array([[l-1, (4*i)+2]])
    t[l-2][(4*i)+2] = np.array([[l-1, (4*i)+3]])
    t[l-2][(4*i)+3] = np.array([[l-1, (4*i)+4]])
    
    if (l-1) > 1:
        f,t = sets(l-1, 4*i, f, t)
        f,t = sets(l-1, (4*i)+1, f, t)
        f,t = sets(l-1, (4*i)+2, f, t)
        f,t = sets(l-1, (4*i)+3, f, t)
    
    return f,t
