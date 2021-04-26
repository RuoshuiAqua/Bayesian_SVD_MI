# Bayesian_SVD_MI

This repo contains code for our proposed Bayesian Singular Value Decomposition (SVD) model for multiple imputation

* Bayesian_lasso_svd includes simulation code for the basic Bayesian Lasso SVD model when the data matrix is fully observed
* Bayesian_lasso_svd_mi include simulation code for the multiple imputation model under 3 difference missing mechanisms
* s-vae-pytorch is a python library which contains a Pytorch implementation of the hyperspherical variational auto-encoder, or S-VAE, as presented in Davidson, Tim R., et al. (http://arxiv.org/abs/1804.00891). It's mainly used for sampling from von Mises-Fisher (vMF) distribution in this repo.