import numpy as np
import torch
import numpy.linalg as linalg
import scipy.stats as stats
import pickle
from sys import exit

m = 20
n = 40
K = 5

np.random.seed(0)

U = stats.ortho_group.rvs(m)[:, 0:K]
V = stats.ortho_group.rvs(n)[:, 0:K]
mu_mn = np.sqrt(m + n + 2*np.sqrt(m*n))
D = np.random.uniform(low = 1/2*mu_mn, high = 3/2*mu_mn, size = K)
D = np.array([16.0, 14.0, 10.0, 2.0, 1.0])
M = np.matmul(U*D, V.T)
E = np.random.normal(size = M.shape)
Y = M + E

with open(f"./output/data_m_{m}_n_{n}_K_{K}.pkl", 'wb') as file_handle:
    pickle.dump({"m": m, 'n': n, 'K': K,
                 'U': U, 'D': D, 'V': V,
                 'M': M, 'Y': Y, 'E': E}, file_handle)
    
## descretize some columns using link function
shuffled_col_index = np.arange(n)
np.random.shuffle(shuffled_col_index)
binary_col_index = shuffled_col_index[0:10]
count_col_index = shuffled_col_index[10:20]
continuous_col_index = shuffled_col_index[20:]

E = np.random.normal(scale = 0.3, size = M.shape)
linked_Y = np.copy(M) + E

p = 1.0 / (1.0 + np.exp(-3*linked_Y[:, binary_col_index]))
linked_Y[:, binary_col_index] = np.random.binomial(n = 1, p = p).astype(np.float64)

lam = np.exp(linked_Y[:, count_col_index])
linked_Y[:, count_col_index] = np.random.poisson(lam).astype(np.float64)

linked_Y[:, continuous_col_index] = linked_Y[:, continuous_col_index]

with open(f"./output/linked_data_m_{m}_n_{n}_K_{K}.pkl", 'wb') as file_handle:
    pickle.dump({"m": m, 'n': n, 'K': K,
                 'U': U, 'D': D, 'V': V,
                 'M': M, 'linked_Y': linked_Y, 'E': E,
                 'binary_col_index': binary_col_index,
                 'count_col_index': count_col_index,
                 'continuous_col_index': continuous_col_index}, file_handle)
