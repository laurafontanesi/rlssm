data {
	int<lower=1> N;									// number of data items
	int<lower=1> K;									// number of options
	int<lower=1> L;									// number of levels
	array[N] int<lower=1, upper=L> participant;			// level (participant)
	array[N] int<lower=1> trial_block;					// trial within block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	array[N] int<lower=1, upper=K> cor_option;			// correct option
	array[N] int<lower=1, upper=K> inc_option;			// incorrect option
	array[N] int<lower=1> block_label;					// block label

	array[N] int<lower=-1,upper=1> accuracy;			// accuracy (-1, 1)
	array[N] real<lower=0> rt;							// rt
	array[N] int<lower=0, upper=1> feedback_type; 		// feedback_type = 0 -> full feedback, feedback_type = 1 -> partial feedback

	real initial_value;								// intial value for learning in the first block

	vector[4] alpha_priors;							// mean and sd of the prior
	vector[4] drift_scaling_priors;					// mean and sd of the prior
	vector[4] threshold_priors;						// mean and sd of the prior
	vector[4] threshold_modulation_priors;			// mean and sd of the prior
	vector[4] ndt_priors;							// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real mu_alpha;
	real mu_drift_scaling;
	real mu_threshold;
	real mu_threshold_modulation;
	real mu_ndt;

	real<lower=0> sd_alpha;
	real<lower=0> sd_drift_scaling;
	real<lower=0> sd_threshold;
	real<lower=0> sd_threshold_modulation;
	real<lower=0> sd_ndt;

	array[L] real z_alpha;
	array[L] real z_drift_scaling;
	array[L] real z_threshold;
	array[L] real z_threshold_modulation;
	array[L] real z_ndt;
}
transformed parameters {
	array[N] real drift_ll;								// trial-by-trial drift rate for likelihood (incorporates accuracy)
	array[N] real drift_t;								// trial-by-trial drift rate for predictions
	array[N] real<lower=0> threshold_t;					// trial-by-trial threshold   
	array[N] real<lower=0> ndt_t;							// trial-by-trial ndt

	vector[K] Q;									// Q state values

	real Q_mean;									// mean across all options
	array[N] real Q_mean_pres;							// mean Q presented options
	array[N] real delta_Q;								// Qcor - Qinc
	real PE_cor;									// predicion error correct option
	real PE_inc;									// predicion error incorrect option

	array[L] real<lower=0, upper=1> alpha_sbj;
	array[L] real<lower=0> drift_scaling_sbj;
	array[L] real threshold_sbj;
	array[L] real threshold_modulation_sbj;
	array[L] real<lower=0> ndt_sbj;

	real transf_mu_alpha;
	real transf_mu_drift_scaling;
	real transf_mu_threshold;
	real transf_mu_threshold_modulation;
	real transf_mu_ndt;

	transf_mu_alpha = Phi(mu_alpha);				// for the output
	transf_mu_drift_scaling = log(1 + exp(mu_drift_scaling));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_threshold_modulation = mu_threshold_modulation;
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
		alpha_sbj[l] = Phi(mu_alpha + z_alpha[l]*sd_alpha);
		drift_scaling_sbj[l] = log(1 + exp(mu_drift_scaling + z_drift_scaling[l]*sd_drift_scaling));
		threshold_sbj[l] = mu_threshold + z_threshold[l]*sd_threshold;
		threshold_modulation_sbj[l] = mu_threshold_modulation + z_threshold_modulation[l]*sd_threshold_modulation;
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		if (trial_block[n] == 1) {
			if (block_label[n] == 1) {
				Q = Q0;
			} else {
				Q_mean = mean(Q);
				Q = rep_vector(Q_mean, K);
			}
		}
		Q_mean_pres[n] = (Q[cor_option[n]] + Q[inc_option[n]])/2;
		delta_Q[n] = Q[cor_option[n]] - Q[inc_option[n]];
		PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];

		drift_t[n] = drift_scaling_sbj[participant[n]]*delta_Q[n];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = log(1 + exp(threshold_sbj[participant[n]] + threshold_modulation_sbj[participant[n]]*Q_mean_pres[n]));
		ndt_t[n] = ndt_sbj[participant[n]];

		if (feedback_type[n] == 1){
      if(accuracy[n] == 1){
        Q[cor_option[n]] = Q[cor_option[n]] + alpha_sbj[participant[n]]*PE_cor;
      }
      else{
        Q[inc_option[n]] = Q[inc_option[n]] + alpha_sbj[participant[n]]*PE_inc;
      }
    }
    else{
      Q[cor_option[n]] = Q[cor_option[n]] + alpha_sbj[participant[n]]*PE_cor;
      Q[inc_option[n]] = Q[inc_option[n]] + alpha_sbj[participant[n]]*PE_inc;
    }

	}
}
model {
	mu_alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	mu_drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_threshold_modulation ~ normal(threshold_modulation_priors[1], threshold_modulation_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);
	sd_drift_scaling ~ normal(drift_scaling_priors[3], drift_scaling_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_threshold_modulation ~ normal(threshold_modulation_priors[3], threshold_modulation_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

	z_alpha ~ normal(0, 1);
	z_drift_scaling ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_threshold_modulation ~ normal(0, 1);
	z_ndt ~ normal(0, 1);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
	}
}