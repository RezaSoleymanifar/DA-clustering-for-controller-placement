# %load testAlg.py
import numpy as np
import sys


#sample input: 'data.npy', 10, 0.05, 3, 'lb'
data, max_m, gamma, interactive, mode = sys.argv[1:]
data = np.load(data)

if mode == 'll':
    from DALL import DA    
elif mode == 'lb':
    from DALB import DA
    
obj = DA(data = data, max_m = int(max_m), gamma = float(gamma), interactive = int(interactive))    
obj.train()
