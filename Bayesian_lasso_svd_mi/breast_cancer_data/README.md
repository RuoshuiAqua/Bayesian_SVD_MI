1. run ``python ./script/simulate_data.py --m 20 --n 40 --K 5`` to simulate data.
2. run ``Rscript ./script/generate_omega.r 20 40 5`` to generate values of the missing indicator variable at three different missing scenarios:
   mcar: missing completely at random
   corr: missing at random with missing probability depends on column-wise correlation of the true matrix M
   sv: missing at random with missing probability depends on largest singular singular value of the true matrix
3. run ``python ./script/compute_prior_parameters_missing.py --m 20 --n 40 --K 5 --missing_type mcar`` to compute parameters of prior distributions using an empirical Bayesian approach for the missing scenario ``mcar``.
   run ``python ./script/compute_prior_parameters_missing.py --m 20 --n 40 --K 5 --missing_type corr`` to compute parameters of prior distributions using an empirical Bayesian approach for the missing scenario ``corr``.
   run ``python ./script/compute_prior_parameters_missing.py --m 20 --n 40 --K 5 --missing_type sv`` to compute parameters of prior distributions using an empirical Bayesian approach for the missing scenario ``sv``.
4. run ``python ./script/gibbs_sampler_with_MCEM_lambda.py --m 20 --n 40 --K 5 --missing_type mcar`` to run Gibbs sampler for missing type mcar. Similar commands can be used for missing types of corr and sv.
5. The estimated Y for missing type mcar from Gibbs sampler is saved as "Y_est" in the file ./output/samples_missing_mcar_20_40_5.pkl.