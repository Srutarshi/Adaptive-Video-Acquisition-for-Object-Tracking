import numpy as np
import math as ma
import argparse
import time
import copy

def qt_segment(Nl):
    Na = Nl
    AN0 = 1
    a = [None for i in range(Na)]
    for n in range(1,Na+1):   # index change than in MATLAB
        a[n-1] = np.zeros((1,4**(Na-n)))

    a[Na-1] = np.array([[AN0]])
    a = segment(Na,0,a)
    return a

def segment(l,i,a):
    a[l-2][0,4*i] = a[l-1][0,i] + 4*(l>2) # index change for l than in MATLAB
    a[l-2][0,(4*i)+1] = 0
    a[l-2][0,(4*i)+2] = 0
    a[l-2][0,(4*i)+3] = 0

    if l-1>1:                    # index change for l than in MATLAB
        a = segment(l-1, 4*i, a)
        a = segment(l-1, (4*i)+1, a)
        a = segment(l-1, (4*i)+2, a)
        a = segment(l-1, (4*i)+3, a)

    return a











