import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import pickle
import argparse
from sys import exit
mpl.rc('font', size = 14)
mpl.rc('axes', titlesize = 'medium', labelsize = 'medium')
mpl.rc('xtick', labelsize = 'medium')
mpl.rc('ytick', labelsize = 'medium')

parser = argparse.ArgumentParser()
parser.add_argument("--m", type = int, default = 20)
parser.add_argument("--n", type = int, default = 40)
parser.add_argument("--K", type = int, default = 5)

args = parser.parse_args()

m = args.m
n = args.n
K = args.K

## read data
with open(f"./output/data_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
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

Uh, Dh, VhT = linalg.svd(Y)
Vh = VhT.T
Uh = Uh[:,0:K]
Dh = Dh[0:K]
Vh = Vh[:,0:K]
Yh_ls = np.matmul(Uh*Dh, Vh.T)
mse_ls = np.mean((Yh_ls - M)**2)
print("MSE of least square: {:.4f}".format(mse_ls))

## data samples from Gibbs sampler of inferred lambda
with open(f"./output/record_with_MCEM_lambda/samples_{m}_{n}_{K}_MCEM_lambda.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    Uh = data['Uh']
    Vh = data['Vh']
    Dh = data['Dh']
    lamb_record = data['lamb']

Yh = []    
for i in range(len(Uh)):
    Yh.append(np.matmul(Uh[i]*Dh[i], Vh[i].T))

Yh = np.array(Yh)
Yh = Yh.mean(0)
mse_inferred_lamb = np.mean((Yh - M)**2)
print("MSE of Bayesian inference lambda: {:.4f}".format(mse_inferred_lamb))


fig = plt.figure(0)
fig.clf()
plt.plot(range(len(lamb_record))[::10],lamb_record[::10], linewidth = 0.5)
plt.xlabel("num of steps in Gibbs sampler")
plt.ylabel(r"$\lambda$")
plt.tight_layout()
plt.savefig(f"./output/MCEM_lambda_m_{m}_n_{n}_K_{K}.eps")

fig = plt.figure(0)
fig.clf()
plt.hist(lamb_record[10000:], 50, density = True)
plt.xlabel(r"$\lambda$")
plt.tight_layout()
plt.savefig(f"./output/MCEM_lambda_hist_m_{m}_n_{n}_K_{K}.eps")

lamb_record = lamb_record[10000:]

## read samples from Gibbs sampler of fixed lambdas
Dh_list = []
num_lambs = 50
mse_list = []
for idx_lamb in range(num_lambs):
    print(idx_lamb)
    with open(f"./output/record_with_fixed_lambda/samples_{m}_{n}_{K}_idx_lambda_{idx_lamb}.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)        
        Uh = data['Uh']
        Vh = data['Vh']
        Dh = data['Dh']
        lambs = data['lambs']
        idx_lamb = data['idx_lamb']

        ## calculate mean squared error
        Yh = []    
        for i in range(len(Uh)):
            Yh.append(np.matmul(Uh[i]*Dh[i], Vh[i].T))
        Yh = np.array(Yh)
        Yh = Yh.mean(0)
        mse_list.append(np.mean((Yh - M)**2))

        ## sort Dh
        Dh = np.abs(Dh)
        Dh = np.sort(Dh)
        Dh = np.mean(Dh[::10], 0)[::-1]

        Dh_list.append(Dh)

Dh = np.array(Dh_list)
mse = np.array(mse_list)

fig = plt.figure(1, figsize = (6.4, 9.6))
plt.subplot(2,1,1)
for i in range(Dh.shape[1]):
    print(i)    
    plt.plot(lambs, Dh[:,i], label = "rank_" + str(i))
#plt.xlim(lambs[-1], lambs[0])    
plt.xscale('log')
#plt.legend()

## observed data
with open(f"./output/data_m_{m}_n_{n}_K_{K}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
D = data['D']
D = np.sort(D)[::-1]
for i in range(len(D)):
    plt.plot(lambs, np.repeat(D[i], len(lambs)) + np.random.normal(scale = 0.02), '--', color = 'k', linewidth = 0.5)

lamb_median = np.quantile(lamb_record, 0.5)
lamb_low = np.quantile(lamb_record, 0.025)
lamb_high = np.quantile(lamb_record, 0.975)
plt.axvline(x = lamb_median, linestyle = '--', color = 'k')
plt.axvline(x = lamb_low, linestyle = '--', color = 'k')
plt.axvline(x = lamb_high, linestyle = '--', color = 'k')
plt.ylim([0,22])
plt.ylabel("singular values")
plt.xticks(ticks = None)
#plt.xlabel(r"$\lambda$")
plt.tight_layout()

plt.subplot(2,1,2)
plt.axhline(y = mse_ls, linestyle = "--", color = "r", label = "Least Square")
plt.plot(lambs, mse, color = 'k', label = r"Bayesian SVD (fixed $\lambda$)")
#plt.xlim(lambs[-1], lambs[0])
plt.xscale('log')
plt.axhline(y = mse_inferred_lamb, linestyle = "--", color = 'b', label = r'Bayesian SVD (MCEM $\lambda$)')
plt.ylim([0.1,1.2])
plt.axvline(x = lamb_median, linestyle = '--', color = 'k')
plt.axvline(x = lamb_low, linestyle = '--', color = 'k')
plt.axvline(x = lamb_high, linestyle = '--', color = 'k')
plt.xlabel(r"$\lambda$")
plt.ylabel("MSE")
plt.legend(fontsize = "small")
plt.tight_layout()
plt.savefig(f"./output/singular_values_m_{m}_n_{n}_K_{K}.eps")
