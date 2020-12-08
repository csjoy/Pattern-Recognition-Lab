import os
import pandas as pd
import numpy as np


file_list = sorted(os.listdir('NPY_New'))
os.mkdir('NPY_Final')

files = list()
ranges = dict()

for files in file_list:
    file_info = files.split('.')

    if files[0]=='N':
        a = np.load(f"NPY_New/{files}")
        b = np.load(f"NPY_New/P{files[1:]}")
        c = np.concatenate((a, b), axis=0)
        
        x, y = a.shape
        zero = np.zeros((x, 1))
        ranges[f"{file_info[0]}"] = tuple([0,x-1])
        z = x
        x, y = b.shape
        one = np.ones((x, 1))
        x, y = c.shape
        
        ranges[f"P{file_info[0][1:]}"] = tuple([z, x-1])
        np.save(f"NPY_Final/X_{file_info[0][1:]}.npy", c)
        print(c.shape)
        d = np.concatenate((zero, one), axis=0)
        np.save(f"NPY_Final/Y_{file_info[0][1:]}.npy", d)

print(ranges)
