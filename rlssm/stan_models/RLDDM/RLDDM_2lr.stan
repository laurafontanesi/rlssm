data {
	int<lower=1> N;									// number of data items
	int<lower=1> K;									// number of options
	array[N] int<lower=1> trial_block;					// trial within block
	vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option
	array[N] int<lower=1, upper=K> cor_option;			// correct option
	array[N] int<lower=1, upper=K> inc_option;			// incorrect option
	array[N] int<lower=1> block_label;					// block label

	array[N] int<lower=-1,upper=1> accuracy;				// accuracy (-1, 1)
	array[N] real<lower=0> rt;							// rt
	array[N] int<lower=0, upper=1> feedback_type; 		// feedback_type = 0 -> full feedback, feedback_type = 1 -> partial feedback

	real initial_value;								// intial value for learning in the first block

	vector[2] alpha_pos_priors;						// mean and sd of the alpha_pos prior
	vector[2] alpha_neg_priors;						// mean and sd of the alpha_neg prior
	vector[2] drift_scaling_priors;					// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
	real<lower=0, upper=1> starting_point;			// starting point diffusion model not to estimate
}
transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}
parameters {
	real alpha_pos;
	real alpha_neg;
	real drift_scaling;
	real threshold;
	real ndt;
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

	real transf_alpha_pos;
	real transf_alpha_neg;
	real transf_drift_scaling;
	real transf_threshold;
	real transf_ndt;

	transf_alpha_pos = Phi(alpha_pos);				// for the output
	transf_alpha_neg = Phi(alpha_neg);
	transf_drift_scaling = log(1 + exp(drift_scaling));
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));

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

		drift_t[n] = transf_drift_scaling*delta_Q[n];
		drift_ll[n] = drift_t[n]*accuracy[n];
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;

		if (feedback_type[n] == 1){
      if(accuracy[n] == 1){
        if (PE_cor >= 0) {
          Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_pos*PE_cor;
        } else {
          Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_neg*PE_cor;
        }
      }
      else{
        if (PE_inc >= 0) {
          Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_pos*PE_inc;
        } else {
          Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_neg*PE_inc;
        }
      }
    }
    else{
      if (PE_cor >= 0) {
        Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_pos*PE_cor;
      } else {
        Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_neg*PE_cor;
      }
      if (PE_inc >= 0) {
        Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_pos*PE_inc;
      } else {
        Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_neg*PE_inc;
      }
    }
    
	}
}
model {
	alpha_pos ~ normal(alpha_pos_priors[1], alpha_pos_priors[2]);
	alpha_neg ~ normal(alpha_neg_priors[1], alpha_neg_priors[2]);
	drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	rt ~ wiener(threshold_t, ndt_t, starting_point, drift_ll);
}
generated quantities {
	vector[N] log_lik;

	{for (n in 1:N) {
		log_lik[n] = wiener_lpdf(rt[n] | threshold_t[n], ndt_t[n], starting_point, drift_ll[n]);
	}
	}
}