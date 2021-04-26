import numpy as np
import torch
import numpy.linalg as linalg
import scipy.stats as stats
import pickle
from sys import exit
import argparse
import os
import pandas as pd
from collections import defaultdict
import sklearn.datasets as datasets

data = datasets.load_breast_cancer()
Y = data.data
flag = data.target == 0
Y = Y[flag][0:50]

m = 50
n = 30
K = 25

Y_mean = np.mean(Y, 0, keepdims = True)
Y_std = np.std(Y, 0, keepdims = True)
Y = (Y - Y_mean)/Y_std

np.random.seed(123)

U = None
V = None
mu_mn = np.sqrt(m + n + 2*np.sqrt(m*n))
D = None
M = Y
E = np.random.normal(loc = 0.0, scale = 0.1, size = M.shape)
Y = M + E

os.makedirs("./output/simulated_data", exist_ok = True)
with open(f"./output/simulated_data/data_m_{m}_n_{n}_K_{K}.pkl", 'wb') as file_handle:
    pickle.dump({"m": m, 'n': n, 'K': K,
                 'U': U, 'D': D, 'V': V,
                 'M': M, 'Y': Y, 'E': E}, file_handle)    
np.savetxt(f"./output/simulated_data/M_m_{m}_n_{n}_K_{K}.csv", M, delimiter = ",")
np.savetxt(f"./output/simulated_data/Y_m_{m}_n_{n}_K_{K}.csv", Y, delimiter = ",")
    
