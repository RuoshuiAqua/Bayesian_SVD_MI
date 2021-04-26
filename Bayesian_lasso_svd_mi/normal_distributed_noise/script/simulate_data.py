import numpy as np
import torch
import numpy.linalg as linalg
import scipy.stats as stats
import pickle
from sys import exit
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--m", type = int, default = 20)
parser.add_argument("--n", type = int, default = 40)
parser.add_argument("--K", type = int, default = 5)
args = parser.parse_args()

m = args.m
n = args.n
K = args.K

np.random.seed(2)

U = stats.ortho_group.rvs(m)[:, 0:K]
V = stats.ortho_group.rvs(n)[:, 0:K]
mu_mn = np.sqrt(m + n + 2*np.sqrt(m*n))
D = np.random.uniform(low = 1/2*mu_mn, high = 3/2*mu_mn, size = K)
M = np.matmul(U*D, V.T)
E = np.random.normal(size = M.shape)
Y = M + E

os.makedirs("./output/simulated_data", exist_ok = True)
with open(f"./output/simulated_data/data_m_{m}_n_{n}_K_{K}.pkl", 'wb') as file_handle:
    pickle.dump({"m": m, 'n': n, 'K': K,
                 'U': U, 'D': D, 'V': V,
                 'M': M, 'Y': Y, 'E': E}, file_handle)    
np.savetxt(f"./output/simulated_data/M_m_{m}_n_{n}_K_{K}.csv", M, delimiter = ",")
np.savetxt(f"./output/simulated_data/Y_m_{m}_n_{n}_K_{K}.csv", Y, delimiter = ",")

    
