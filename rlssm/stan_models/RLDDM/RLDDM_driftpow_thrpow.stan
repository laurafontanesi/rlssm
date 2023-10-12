data {
	int<lower=1> N;									// number of data items
	int<lower=1> K;									// number of options
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

	vector[2] alpha_priors;							// mean and sd of the prior
	vector[2] drift_scaling_priors;					// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
    vector[2] drift_power_priors;					// mean and sd of the prior
    vector[2] threshold_power_priors;				// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}

transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}

parameters {
	real alpha;
	real drift_scaling;
	real threshold;
	real ndt;
    real drift_power;
    real threshold_power;
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
	real PE_cor;									// prediction error correct option
	real PE_inc;									// prediction error incorrect option

	real transf_alpha;
	real transf_drift_scaling;
	real transf_threshold;
	real transf_ndt;
    real transf_drift_power;
    real transf_threshold_power;

	transf_alpha = Phi(alpha);						// for the output
	transf_drift_scaling = log(1 + exp(drift_scaling));
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));
    transf_drift_power = drift_power;
    transf_threshold_power = threshold_power;

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

		drift_t[n] = transf_drift_scaling * delta_Q[n] * pow(trial_block[n]/10.0, transf_drift_power);
		drift_ll[n] = drift_t[n] * accuracy[n];
		threshold_t[n] = transf_threshold * pow(trial_block[n]/10.0, transf_threshold_power);
		ndt_t[n] = transf_ndt;

		if (feedback_type[n] == 1){
      if(accuracy[n] == 1){
        Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha*PE_cor;
      }
      else{
        Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha*PE_inc;
      }
    }
    else{
      Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha*PE_cor;
      Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha*PE_inc;
    }

	}
}

model {
	alpha ~ normal(alpha_priors[1], alpha_priors[2]);
	drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);
    drift_power ~ normal(drift_power_priors[1], drift_power_priors[2]);
    threshold_power ~ normal(threshold_power_priors[1], threshold_power_priors[2]);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}

generated quantities {
	vector[N] log_lik;

	{
        for (n in 1:N) {
		    log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	    }
	}
}
