import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import numpy.linalg as linalg
import scipy.stats as stats
from scipy.linalg import null_space
import pickle
from sys import exit
import sys
sys.path.append("../s-vae-pytorch")
from hyperspherical_vae.distributions import *
import os
import argparse
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("--m", type = int, default = 20)
parser.add_argument("--n", type = int, default = 40)
parser.add_argument("--K", type = int, default = 5)
parser.add_argument("--idx_lambda", type = int,
                    help = 'Choose the index of lambda to use. It should be an integer between 0 and 49, where idx_lambda = 0 cooresponds to lambda = 0.01 and idx_lambda = 49 corresponds to lambda = 100.')

args = parser.parse_args()

m = args.m
n = args.n
K = args.K
idx_lambda = args.idx_lambda

# print(os.environ['SLURM_ARRAY_TASK_ID'])
# idx_lamb = int(os.environ['SLURM_ARRAY_TASK_ID'])
lambs = np.logspace(start = -2, stop = 2, num = 50)
lamb = lambs[idx_lambda]

## parameters for prior distribution 
with open(f"./output/prior_distribution_parameters_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
nu0 = data['nu0']
sigma0_square = data['sigma0_square']

## observed data
with open(f"./output/data_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
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
phi = 1./sigma0_square
Uh, Dh, VhT = linalg.svd(Y)
Vh = VhT.T

K = min(m,n)-2
Uh = Uh[:,0:K]
Dh = Dh[0:K]
Vh = Vh[:,0:K]

tau_square = stats.expon.rvs(scale = 2./lamb**2, size = K)

Uh_record = []
Vh_record = []
Dh_record = []

num_cycles = 20000

for idx_cycle in range(num_cycles):
    if (idx_cycle + 1) % 10 == 0:
        print(f"idx of cycle: {idx_cycle}", flush = True)

    ## sample phi, mu, and psi
    res = Y - np.matmul(Uh*Dh, Vh.T)
    phi = stats.gamma.rvs(a = (nu0+m*n+K)/2.,
                          scale = 2./(nu0*sigma0_square + np.sum(res**2) + np.sum(Dh**2/tau_square)))

    # the definition of scale in stats.invgauss is very unintuitive
    scale = lamb**2    
    inv_tau_square = stats.invgauss.rvs(mu = np.sqrt(lamb**2/(Dh**2*phi))/scale,
                                        scale = scale)
    tau_square = 1./inv_tau_square
    
    ## sample U, V and D
    for j in range(K):
        index_mask = np.ones(K, dtype = np.bool)
        index_mask[j] = False
        Eminusj = Y - np.matmul(Uh[:, index_mask]*Dh[index_mask], Vh[:, index_mask].T)

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
                              
                              
Yh_record = []
for i in range(len(Uh_record)):
    print(i)
    Yh_record.append(np.matmul(Uh_record[i]*Dh_record[i], Vh_record[i].T))

Yh = np.array(Yh_record)
Yh = Yh.mean(0)

print("Bayesian mean of squared erro: {:.4f}".format(np.mean((Yh - M)**2)))

K = K_true
Uh, Dh, VhT = linalg.svd(Y)
Vh = VhT.T
Uh = Uh[:,0:K]
Dh = Dh[0:K]
Vh = Vh[:,0:K]
Yh_ls = np.matmul(Uh*Dh, Vh.T)

print("least square mean of squared erro: {:.4f}".format(np.mean((Yh_ls - M)**2)))

os.makedirs(f"./output/record_with_fixed_lambda", exist_ok = True)
with open(f"./output/record_with_fixed_lambda/samples_{m}_{n}_{K_true}_idx_lambda_{idx_lambda}.pkl", 'wb') as file_handle:
    pickle.dump({'Uh': np.array(Uh_record), 'Vh': np.array(Vh_record), 'Dh': np.array(Dh_record),
                 'lambs': lambs, 'idx_lamb': idx_lambda}, file_handle)
    
