import numpy as np
import math as ma
import argparse
import time
import copy

def qt_orient(Nl):
    No = Nl - 1                     # variable ON0 is not used here as in MATLAB
    ON0 = 0
    o = [None for i in range(No)]
    for n in range(0,No):
        a = int(ma.pow(4,No-n-1))
        o[n] = np.zeros((a))
    o[No-1] = np.array([ON0])
    o = orient(No-1,0,o)
    #print (o)
    return o

def orient(l,i,o):
    oli = o[l][i]                    # the index changes here than in MATLAB
    o[l-1][4*i] = (5-oli)%4          # the index changes here than in MATLAB
    o[l-1][(4*i)+1] = oli            # the index changes here than in MATLAB
    o[l-1][(4*i)+2] = oli            # the index changes here than in MATLAB
    o[l-1][(4*i)+3] = 3 - oli        # the index changes here than in MATLAB
    
    if (l-1) > 0:                    # change in index here than in MATLAB
        o = orient(l-1, 4*i, o)
        o = orient(l-1, (4*i)+1, o)
        o = orient(l-1, (4*i)+2, o)
        o = orient(l-1, (4*i)+3, o)
    
    return o










