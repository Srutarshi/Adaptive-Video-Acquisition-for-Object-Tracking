from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy

def qt_distortion(v,stdev):
    Nv = len(v)
    d = [None for i in range(Nv)]
    for n in range(0,Nv):
        if n<3:
            v[n] = np.Infinity*(np.ones(v[n].shape))
     
        a1 = np.multiply(v[n],v[n]>2)
        d[n] = np.vstack([a1, dist_when_acquire(stdev[n],Nv,n)])
    return d

def dist_when_acquire(s,N,n):
    th = 3                            # changed from 3
    if (N-n-1)>th:                    # not necessary
        s = np.Infinity*(s+1)     # Do not acquire big blocks (threshold depending on bitrate)
    else:
        s = s*(ma.pow(4,N-n-1))     # 0 for finest level(1 px blocks), cos s will be 0
    s1 = (s>2)
    s = np.multiply(s,s1)      # if zero distortion, then not a priority

    return s

