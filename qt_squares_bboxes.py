from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy

def qt_squares_bboxes(Nl,o):
    sq = [None for i in range(Nl)]
    Nv = int(ma.pow(2,2*(Nl-1)))
    sq[Nl-1] = np.zeros((2,Nv))
    #print(sq[9].shape)
    #print(len(sq))

    pm = int(ma.pow(2,(Nl-1)))
    pn = 1
    n = 1
    lon = len(o[n-1])
    #print ('lon=',lon)
    i = 1
    
    sq[Nl-1][:,i-1] = [pm,pn] # The command is equal to that in MATLAB line 21
    #print(sq[Nl-1][:,i-1])
    while i<lon:
        j = (4*i)-4
        #print (i,j)
        if o[n-1][i-1] == 0:
            sq[Nl-1][:,j] = [pm,pn]
            sq[Nl-1][:,j+1] = [pm,pn+1]
            sq[Nl-1][:,j+2] = [pm-1,pn+1]
            sq[Nl-1][:,j+3] = [pm-1,pn]
            pm = pm - 1
        if o[n-1][i-1] == 1:
            sq[Nl-1][:,j] = [pm,pn]
            sq[Nl-1][:,j+1] = [pm-1,pn]
            sq[Nl-1][:,j+2] = [pm-1,pn+1]
            sq[Nl-1][:,j+3] = [pm,pn+1]
            pn = pn + 1
        if o[n-1][i-1] == 2:
            sq[Nl-1][:,j] = [pm,pn]
            sq[Nl-1][:,j+1] = [pm,pn-1]
            sq[Nl-1][:,j+2] = [pm+1,pn-1]
            sq[Nl-1][:,j+3] = [pm+1,pn]
            pm = pm + 1
        if o[n-1][i-1] == 3:
            sq[Nl-1][:,j] = [pm,pn]
            sq[Nl-1][:,j+1] = [pm+1,pn]
            sq[Nl-1][:,j+2] = [pm+1,pn-1]
            sq[Nl-1][:,j+3] = [pm,pn-1]
            pn = pn - 1

        r = i%4
        ni = n+1
        k = 1
        while r==0:
                a = int(ma.pow(4,k))
                b = int(i/a)
                r = b%4
                k = k + 1
                ni = ni + 1

        b1 = int(ma.pow(4,k))
        a1 = int(ma.ceil(i/b1))
        ori = o[ni-1][a1-1]
        #print(ori)
        if ori == 0:
            shift = (r==1)*[0,1] + (r==2)*[1,0] + (r==3)*[0,-1]
        if ori == 1:
            shift = (r==1)*[1,0] + (r==2)*[0,1] + (r==3)*[-1,0]
        if ori == 2:
            shift = (r==1)*[0,-1] + (r==2)*[-1,0] + (r==3)*[0,1]
        if ori == 3:
            shift = (r==1)*[-1,0] + (r==2)*[0,-1] + (r==3)*[1,0]

        #print(shift)
        pm = pm - shift[0]
        pn = pn + shift[1]
        i = i + 1

    j = (4*i) - 4
    if o[n-1][i-1] == 0:
        sq[Nl-1][:,j] = [pm,pn]
        sq[Nl-1][:,j+1] = [pm,pn+1]
        sq[Nl-1][:,j+2] = [pm-1,pn+1]
        sq[Nl-1][:,j+3] = [pm-1,pn]

    if o[n-1][i-1] == 1:
        sq[Nl-1][:,j] = [pm,pn]
        sq[Nl-1][:,j+1] = [pm-1,pn]
        sq[Nl-1][:,j+2] = [pm-1,pn+1]
        sq[Nl-1][:,j+3] = [pm,pn+1]

    if o[n-1][i-1] == 2:
        sq[Nl-1][:,j] = [pm,pn]
        sq[Nl-1][:,j+1] = [pm,pn-1]
        sq[Nl-1][:,j+2] = [pm+1,pn-1]
        sq[Nl-1][:,j+3] = [pm+1,pn]

    if o[n-1][i-1] == 3:
        sq[Nl-1][:,j] = [pm,pn]
        sq[Nl-1][:,j+1] = [pm+1,pn]
        sq[Nl-1][:,j+2] = [pm+1,pn-1]
        sq[Nl-1][:,j+3] = [pm,pn-1]

    sq[Nl-1] = np.tile(sq[Nl-1],(2,1))  # we repeat the whole sq matrix twice & convert to numpy array
    #print(sq[Nl-1][:,0:12])

    for n in range(Nl-1,0,-1):
        a = int(ma.pow(4,Nl-n))
        nv = int(Nv/a)
        sq[n-1] = np.zeros((4,nv))
        for i in range(0,nv):
            a2 = sq[n][0:2,4*i:4*i+3]
            b2 = sq[n][2:4,4*i:4*i+3]
            sq[n-1][0:2,i] = np.amin(a2, axis=1)
            sq[n-1][2:4,i] = np.amax(b2, axis=1)

    return sq
