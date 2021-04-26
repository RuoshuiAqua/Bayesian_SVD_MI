Bayesian lasso svd without missing data

1. run ``python ./script/simulation_data.py`` to generated simulated data.
2. run ``python ./script/compute_prior_parameters`` to compute parameters for prior distirbutions using an empirical Bayesian approach
3. run ``python ./script/gibbs_sampler_with_fixed_lambda.py --idx_lambda 0`` to run Gibbs sampler for fixed lambda. The value of lambda varies from 0.01 to 100. When idx_lambda = 0, the value of lambda is set to 0.01. When idx_lambda = 49, the value of lambda is set to 100. When idx_lambda is between 0 and 49, the value of lambda is between 0.01 and 100. Here idx_lambda has to be between 0 and 49 inclusively.
4. run ``python ./script/gibbs_sampler_with_MCEM_lambda.py`` to run Gibbs sampler that also samples lambda.
5. run ``python ./script/plot_result.py`` to plot the figure that compares the error from least square fitting, gibbs_sampler_with_fixed_lambda at different lambda values and gibbs_sampler_with_MCEM_lambda.

   
