import numpy as np
import numpy.linalg as linalg
import pickle
import argparse
from sys import exit
import os

parser = argparse.ArgumentParser()
parser.add_argument("--m", type = int, default = 20)
parser.add_argument("--n", type = int, default = 40)
parser.add_argument("--K", type = int, default = 5)
parser.add_argument("--missing_type", type = str)
args = parser.parse_args()

m = args.m
n = args.n
K = args.K
missing_type = args.missing_type

## missing flag
flag_missing = np.loadtxt(f"./output/omega/omega_{missing_type}_m_{m}_n_{n}_K_{K}.csv", delimiter = ",")
flag_missing = flag_missing.astype(np.bool)

## read data
with open(f"./output/simulated_data/data_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)

assert(m == data['m'])
assert(n == data['n'])
assert(K == data['K'])

U = data['U']
D = data['D']
V = data['V']

M = data['M']
Y = data['Y']
E = data['E']

## calculate parameters for prior distribution
nu0 = 2
for j in range(n):
    Y[flag_missing[:,j],j] = np.mean(Y[~flag_missing[:,j], j])

Uh, Dh, VhT = linalg.svd(Y)
Vh = VhT.T

sigma_square_list = []
for k in range(1, min(m, n)):
    Yh = np.matmul(Uh[:,0:k]*Dh[0:k], Vh[:,0:k].T)
    sigma_square_list.append(np.mean((Y-Yh)**2))

sigma0_square = np.mean(sigma_square_list)    
lamb0 = min(m,n)*np.sqrt(sigma0_square)/np.sum(np.abs(Dh))

os.makedirs("./output/prior", exist_ok = True)
with open(f"./output/prior/prior_distribution_parameters_{missing_type}_m_{m}_n_{n}_K_{K}.pkl", 'wb') as file_handle:
    pickle.dump({'nu0': nu0, 'sigma0_square': sigma0_square, 'lamb0': lamb0}, file_handle)
