import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import numpy.linalg as linalg
import scipy.stats as stats
from scipy.linalg import null_space
import pickle
from sys import exit
import sys
sys.path.append("../../s-vae-pytorch")
from hyperspherical_vae.distributions import *
import os
import argparse
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--m", type = int)
parser.add_argument("--n", type = int)
parser.add_argument("--K", type = int)
parser.add_argument("--df", type = int)
parser.add_argument("--missing_type", type = str)
args = parser.parse_args()

m = args.m
n = args.n
K = args.K
df = args.df
missing_type = args.missing_type

## missing flag
flag_missing = np.loadtxt(f"./output/omega/omega_{missing_type}_df_{df}_m_{m}_n_{n}_K_{K}.csv", delimiter = ",")
flag_missing = flag_missing.astype(np.bool)

with open(f"./output/prior/prior_distribution_parameters_{missing_type}_df_{df}_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)

nu0 = data['nu0']
sigma0_square = data['sigma0_square']
lamb = data['lamb0']

## observed data
with open(f"./output/simulated_data/data_df_{df}_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    
m = data['m']
n = data['n']
K_true = data['K']

U = data['U']
D = data['D']
V = data['V']

M = data['M']
Y = data['Y']
E = data['E']

## Gibbs sampler
# fill missing positions with column means
Yh = np.copy(Y)
for j in range(n):
    Yh[flag_missing[:,j],j] = np.mean(Yh[~flag_missing[:,j], j])

phi = 1./sigma0_square
Uh, Dh, VhT = linalg.svd(Yh)
Vh = VhT.T

K1 = min(m,n)-2
K2 = K_true + 5
K = min(K1, K2)


Uh = Uh[:,0:K]
Dh = Dh[0:K]
Vh = Vh[:,0:K]

tau_square = stats.expon.rvs(scale = 2./lamb**2, size = K)

Uh_record = []
Vh_record = []
Dh_record = []
lamb_record = []

num_cycles = 10000

Y_est = 0
n_est = 0

np.random.seed(12)
torch.manual_seed(12)

for idx_cycle in range(num_cycles):
    print(f"idx of cycle: {idx_cycle}, lambda: {lamb:.4f}", flush = True)

    ## sample phi, mu, and psi
    res = Yh - np.matmul(Uh*Dh, Vh.T)
    phi = stats.gamma.rvs(a = (nu0+m*n+K)/2.,
                          scale = 2./(nu0*sigma0_square + np.sum(res**2) + np.sum(Dh**2/tau_square)))

    # the definition of scale in stats.invgauss is very unintuitive
    scale = lamb**2    
    inv_tau_square = stats.invgauss.rvs(mu = np.sqrt(lamb**2/(Dh**2*phi))/scale,
                                        scale = scale)
    tau_square = 1./inv_tau_square

    lamb = np.sqrt(2*K/np.sum(tau_square))
    lamb_record.append(lamb)

    if lamb >= 1000:
        exit()
    
    ## sample U, V and D
    for j in range(K):
        index_mask = np.ones(K, dtype = np.bool)
        index_mask[j] = False
        Eminusj = Yh - np.matmul(Uh[:, index_mask]*Dh[index_mask], Vh[:, index_mask].T)

        # sample U[:,j]
        Nminusj = null_space(Uh[:, index_mask].T)    
        direction = phi*Dh[j]*np.matmul(Nminusj.T, np.matmul(Eminusj, Vh[:,j]))
        kappa = np.sqrt(np.sum(direction**2))
        direction = direction / kappa

        vmf = VonMisesFisher(loc = torch.from_numpy(direction),
                             scale = torch.tensor([kappa]))
        uj = vmf.sample().numpy()        
        uj = np.matmul(Nminusj, uj)
        Uh[:,j] = uj

        # sample V[:,j]    
        Nminusj = null_space(Vh[:, index_mask].T)
        direction = phi*Dh[j]*np.matmul(Uh[:,j], np.matmul(Eminusj, Nminusj))
        kappa = np.sqrt(np.sum(direction**2))
        direction = direction / kappa
        
        vmf = VonMisesFisher(loc = torch.from_numpy(direction),
                             scale = torch.tensor([kappa]))
        vj = vmf.sample().numpy()
        
        vj = np.matmul(Nminusj, vj)
        Vh[:,j] = vj

        # sample D[j,j]        
        dj = np.random.normal(loc = np.matmul(Uh[:,j], np.matmul(Eminusj, Vh[:,j]))/(1. + 1./tau_square[j]),
                              scale = np.sqrt(1/(phi + phi/tau_square[j])))
        
        Dh[j] = dj

    if idx_cycle >= num_cycles//2:
        Uh_record.append(np.copy(Uh))
        Vh_record.append(np.copy(Vh))
        Dh_record.append(np.copy(Dh))
        
        n_est = n_est + 1
        Y_est = Y_est + (np.matmul(Uh*Dh, Vh.T) - Y_est)/n_est
        
    ## sample missing positions
    Yh = np.matmul(Uh*Dh, Vh.T) + np.random.normal(scale = np.sqrt(1./phi), size = Yh.shape)
    Yh[~flag_missing] = Y[~flag_missing]

    
# Yh_record = []
# for i in range(len(Uh_record)):
#     print(i)
#     Yh_record.append(np.matmul(Uh_record[i]*Dh_record[i], Vh_record[i].T))

# Yh = np.array(Yh_record)
# Yh = Yh.mean(0)

print("Bayesian mean of squared erro: {:.4f}".format(np.mean((Y_est - M)**2)))

# fill missing positions with column means
Yh = np.copy(Y)
for j in range(n):
    Yh[flag_missing[:,j],j] = np.mean(Yh[~flag_missing[:,j], j])

K = K_true
Uh, Dh, VhT = linalg.svd(Yh)
Vh = VhT.T
Uh = Uh[:,0:K]
Dh = Dh[0:K]
Vh = Vh[:,0:K]
Yh_ls = np.matmul(Uh*Dh, Vh.T)

print("least square mean of squared erro: {:.4f}".format(np.mean((Yh_ls - M)**2)))

os.makedirs("./output/record", exist_ok = True)
with open(f"./output/record/samples_missing_{missing_type}_{df}_{m}_{n}_{K_true}.pkl", 'wb') as file_handle:
    pickle.dump({'Uh': np.array(Uh_record), 'Vh': np.array(Vh_record), 'Dh': np.array(Dh_record),
                 'lamb': np.array(lamb_record), 'Y_est': Y_est}, file_handle)
    
    
