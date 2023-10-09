from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy
from qt_values import *
from qt_distortion import *

def qt_viterbi_segm(x,xpred,N,n0,o,a,f,t,segmentation,p,area_p):
    Nl = N - n0 + 1
    diff = abs(x-xpred)
    lamb = 650
    v = qt_values(diff,N,n0,o,'sum')
    stdev = qt_values(xpred,N,n0,o,'std')
    d = qt_distortion(v,stdev)	

    area_total = np.multiply(512,512)
    area_rest = area_total - area_p
    #factor_p = np.divide(area_p,area_total)
    #factor_rest = np.divide(area_rest,area_total)     

    [sz0,sz1] = np.shape(segmentation)
    seg = np.ones((sz0,sz1))

    seg[segmentation == 2] = np.divide(1e7,area_p)
    seg[segmentation == 1] = np.divide(1e6,area_rest)
    pr = qt_values(seg,N,n0,o,'mean')

    gx = [None for i in range(Nl)]
    inn = [None for i in range(Nl)]  #inn replaces in in MATLAB line 19
    rate = [None for i in range(Nl)]
    for k in range(1,Nl+1):
        gx[k-1] = np.inf*(np.ones((4**(Nl-k),2)))
        inn[k-1] = np.zeros((4**(Nl-k),3))
        rate[k-1] = np.zeros((4**(Nl-k),2))

    #  Initialization
    Nt = t[0][0].shape[0]
    for n in range(1,Nt+1):
        level = t[0][0][n-1,0]
        child = t[0][0][n-1,1]
        d11 = d[Nl-level][:,child-1]
        d12 = np.hstack([pr[Nl-level][0,child-1],6])
        temp12 = (d11.T)*d12
        gx[level-1][child-1,:] = temp12 + lamb*(a[level-1][0,child-1] + np.array([0,8]))
        rate[level-1][child-1,:] = a[level-1][0,child-1] + np.array([0,8])

    # correct until here

    #   Recursion
    for k in range(2,4**(N-n0)+1):
        tk = t[0][k-1]
        Nt = tk.shape[0]
        
        for n in range(1,Nt+1):
            level = tk[n-1,0]
            child = tk[n-1,1]

            d11 = d[Nl-level][:,child-1]
            d12 = np.hstack([pr[Nl-level][0,child-1],6])
            temp12 = (d11.T)*d12
            gxn = temp12 + lamb*(a[level-1][0,child-1] + np.array([0,8]))

            fprev = f[0][k-2]
            Nf = fprev.shape[0]
            in_min = np.hstack([fprev[0,:],0])
            gxf_min = np.nanmin(gx[fprev[0,0]-1][fprev[0,1]-1,:])
            in_min[2] = np.nanargmin(gx[fprev[0,0]-1][fprev[0,1]-1,:]) + 1

            for j in range(2,Nf+1):
                gxf = gx[fprev[j-1,0]-1][fprev[j-1,1]-1,:]
                if gxf[0] < gxf_min:
                    gxf_min = gxf[0]
                    in_min = np.hstack([fprev[j-1,:], 1])
                if gxf[1] < gxf_min:
                    gxf_min = gxf[1]
                    in_min = np.hstack([fprev[j-1,:], 2])

            gx[level-1][child-1,:] = gxf_min + gxn #1x1 + 1x2
            inn[level-1][child-1,:] = in_min
            rate[level-1][child-1,:] = a[level-1][0,child-1] + np.array([0,8])

    # Termination
    fprev = f[0][k-1]
    Nf = fprev.shape[0]
    in_min = np.hstack([fprev[0,:], 0])
    gxf_min = np.nanmin(gx[fprev[0,0]-1][fprev[0,1]-1,:])
    in_min[2] = np.nanargmin(gx[fprev[0,0]-1][fprev[0,1]-1,:]) + 1

    for j in range(2,Nf+1):
        gxf = gx[fprev[j-1,0]-1][fprev[j-1,1]-1,:]
        if gxf[0] < gxf_min:
            gxf_min = gxf[0]
            in_min = np.hstack([fprev[j-1,:], 1])
        if gxf[1] < gxf_min:
            gxf_min = gxf[1]
            in_min = np.hstack([fprev[j-1,:], 2])

    #print('gx=',gx)
    #print('in_min=',in_min)

    # Backtracking
    ab = int(ma.pow(4,N))
    ab1 = np.zeros((ab-1,3))
    tree = np.vstack([in_min, ab1])
    tt = 1
    total_rate = 0
    while tree[tt-1,2] > 0:
        tree[tt,:] = inn[int(tree[tt-1,0])-1][int(tree[tt-1,1])-1,:]
        total_rate = total_rate + rate[int(tree[tt-1,0])-1][int(tree[tt-1,1])-1,int(tree[tt-1,2])-1]
        tt = tt+1

    tree = tree[0:tt-1,:]

    return [tree, gxf_min, total_rate]
