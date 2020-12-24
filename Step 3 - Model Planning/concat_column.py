import os
import pandas as pd
import numpy as np


file_list = sorted(os.listdir('NPY'))
os.mkdir('NPY_New')

files = list()
ranges = dict()

x = 0
while x < len(file_list):
    files.append(file_list[x:x+16])
    x = x + 16

for features in files:
    flag = True
    for feat in features:
        feat_split = feat.split('_')
        feat_name = feat_split[0]+"_"+feat_split[1]
        if flag:
            a = np.load(f"NPY/{feat}")
            x = 0
            y, z = a.shape
            flag = False
            ranges[f'{feat}'] = tuple([x, z-1])
        else:
            b = np.load(f"NPY/{feat}")
            a = np.concatenate((a, b), axis=1)
            x = z
            y, z = a.shape
            ranges[f'{feat}'] = tuple([x, z-1])
    print(a.shape)
    np.save(f"NPY_New/{feat_name}.npy", a)

print(ranges)