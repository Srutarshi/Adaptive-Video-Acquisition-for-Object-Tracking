from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy

def qt_values(I,N,n0,o,method):
    Nl = N-n0+1
    v = [None for i in range(Nl)]
    Nv = 2**(2*(Nl-1))
    v[Nl-1] = np.zeros((1,Nv))

    pm = 2**(N-n0)
    pn = 1
    n = 1
    lon = len(o[n-1])
    i = 1
    #print('I shape =',I.shape)
    while i < lon:
        j = (4*i) - 3
        if o[n-1][i-1] == 0:
            v[Nl-1][0,j-1] = I[pm-1,pn-1]
            v[Nl-1][0,j] = I[pm-1,pn]
            v[Nl-1][0,j+1] = I[pm-2,pn]
            v[Nl-1][0,j+2] = I[pm-2,pn-1]
            pm = pm-1

        elif o[n-1][i-1] == 1:
            v[Nl-1][0,j-1] = I[pm-1,pn-1]
            v[Nl-1][0,j] = I[pm-2,pn-1]
            v[Nl-1][0,j+1] = I[pm-2,pn]
            v[Nl-1][0,j+2] = I[pm-1,pn]
            pn = pn+1

        elif o[n-1][i-1] == 2:
            v[Nl-1][0,j-1] = I[pm-1,pn-1]
            v[Nl-1][0,j] = I[pm-1,pn-2]
            v[Nl-1][0,j+1] = I[pm,pn-2]
            v[Nl-1][0,j+2] = I[pm,pn-1]
            pm = pm+1

        elif o[n-1][i-1] == 3:
            v[Nl-1][0,j-1] = I[pm-1,pn-1]
            v[Nl-1][0,j] = I[pm,pn-1]
            v[Nl-1][0,j+1] = I[pm,pn-2]
            v[Nl-1][0,j+2] = I[pm-1,pn-2]
            pn = pn-1

        r = i%4
        ni = n+1
        k = 1
        while r == 0:
            r = int(i/(4**k))%4
            k = k + 1
            ni = ni + 1

        ori = o[ni-1][int(np.ceil(i/(4**k)))-1]

        if ori == 0:
            shift = (r==1)*np.array([0,1]) + (r==2)*np.array([1,0]) + (r==3)*np.array([0,-1])
        elif ori == 1:
            shift = (r==1)*np.array([1,0]) + (r==2)*np.array([0,1]) + (r==3)*np.array([-1,0])
        elif ori == 2:
            shift = (r==1)*np.array([0,-1]) + (r==2)*np.array([-1,0]) + (r==3)*np.array([0,1])
        elif ori == 3:
            shift = (r==1)*np.array([-1,0]) + (r==2)*np.array([0,-1]) + (r==3)*np.array([1,0])

        pm = pm - shift[0]
        pn = pn + shift[1]
        i = i + 1
        
    j = (4*i) - 3
    if o[n-1][i-1] == 0:
        v[Nl-1][0,j-1] = I[pm-1,pn-1]
        v[Nl-1][0,j] = I[pm-1,pn]
        v[Nl-1][0,j+1] = I[pm-2,pn]
        v[Nl-1][0,j+2] = I[pm-2,pn-1]
    
    elif o[n-1][i-1] == 1:
        v[Nl-1][0,j-1] = I[pm-1,pn-1]
        v[Nl-1][0,j] = I[pm-2,pn-1]
        v[Nl-1][0,j+1] = I[pm-2,pn]
        v[Nl-1][0,j+2] = I[pm-1,pn]
    
    elif o[n-1][i-1] == 2:
        v[Nl-1][0,j-1] = I[pm-1,pn-1]
        v[Nl-1][0,j] = I[pm-1,pn-2]
        v[Nl-1][0,j+1] = I[pm,pn-2]
        v[Nl-1][0,j+2] = I[pm,pn-1]
    
    elif o[n-1][i-1] == 3:
        v[Nl-1][0,j-1] = I[pm-1,pn-1]
        v[Nl-1][0,j] = I[pm,pn-1]
        v[Nl-1][0,j+1] = I[pm,pn-2]
        v[Nl-1][0,j+2] = I[pm-1,pn-2]

    if method == 'sum':
        for n in range(Nl-1,0,-1):
            nv = int(Nv/(4**(Nl-n)))
            v[n-1] = np.zeros((1,nv))
            for i in range(1,nv+1):
                v[n-1][0,i-1] = np.sum(v[n][0,4*i-4:4*i])

    elif method == 'mean':
        for n in range(Nl-1,0,-1):
            nv = int(Nv/(4**(Nl-n)))
            v[n-1] = np.zeros((1,nv))
            for i in range(1,nv+1):
                v[n-1][0,i-1] = np.sum(v[n][0,4*i-4:4*i])/4

    elif method == 'std':
        for n in range(Nl-1,0,-1):
            nv = int(Nv/(4**(Nl-n)))
            v[n-1] = np.zeros((1,nv))
            for i in range(1,nv+1):
                K = 4**(Nl-n)
                v[n-1][0,i-1] = np.std(v[Nl-1][0,K*(i-1):K*i])
        v[Nl-1] = v[Nl-1]*0
    
    return v








