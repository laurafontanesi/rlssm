functions {
     real race_pdf(real t, real b, real v){
          real pdf;
          pdf = b/sqrt(2 * pi() * pow(t, 3)) * exp(-pow(v*t-b, 2) / (2*t));
          return pdf;
     }

     real race_cdf(real t, real b, real v){
          real cdf;
          cdf = Phi((v*t-b)/sqrt(t)) + exp(2*v*b) * Phi(-(v*t+b)/sqrt(t));
          return cdf;
     }

     real race_lpdf(matrix RT, vector  ndt, vector b, vector drift_cor, vector drift_inc){

          real t;
          vector[rows(RT)] prob;
          real cdf;
          real pdf;
          real out;

          for (i in 1:rows(RT)){
               t = RT[i,1] - ndt[i];
               if(t > 0){
                  if(RT[i,2] == 1){
                    pdf = race_pdf(t, b[i], drift_cor[i]);
                    cdf = 1 - race_cdf(t, b[i], drift_inc[i]);
                  }
                  else{
                    pdf = race_pdf(t, b[i], drift_inc[i]);
                    cdf = 1 - race_cdf(t, b[i], drift_cor[i]);
                  }
                  prob[i] = pdf*cdf;

                if(prob[i] < 1e-10){
                    prob[i] = 1e-10;
                }
               }
               else{
                    prob[i] = 1e-10;
               }
          }
          out = sum(log(prob));
          return out;
     }
}

data{
  int<lower=1> N;               // number of data items
	int<lower=1> L;								// number of participants
  int<lower=1> K;               // number of total options
  int<lower=1, upper=L> participant[N];			// level (participant)

  real initial_value;

  int<lower=1> block_label[N];					// block label
  int<lower=1> trial_block[N];					// trial within block

  vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option


  int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
  int<lower=1, upper=2> accuracy[N];				// accuracy (1->cor, 2->inc)

  real<lower=0> rt[N];							// reaction time

  vector[4] alpha_pos_priors;						// mean and sd of the mu_alpha_pos prior and sd_alpha_pos prior
	vector[4] alpha_neg_priors;						// mean and sd of the mu_alpha_neg prior and sd_alpha_neg prior
  vector[4] utility_priors;
	vector[4] drift_scaling_priors;			// mean and sd of the prior for scaling
	vector[4] threshold_priors;					// mean and sd of the prior for threshold
	vector[4] ndt_priors;							  // mean and sd of the prior for non-decision time
}

transformed data {
	vector[K] Q0;
  matrix [N, 2] RT;

  Q0 = rep_vector(initial_value, K);

  for (n in 1:N){
    RT[n, 1] = rt[n];
    RT[n, 2] = accuracy[n];
  }
}

parameters {
  real mu_alpha_pos;
	real mu_alpha_neg;
	real mu_drift_scaling;
	real mu_threshold;
	real mu_ndt;
  real mu_utility;

  real<lower=0> sd_alpha_pos;
	real<lower=0> sd_alpha_neg;
	real<lower=0> sd_drift_scaling;
	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;
  real<lower=0> sd_utility;

  real z_alpha_pos[L];
	real z_alpha_neg[L];
	real z_drift_scaling[L];
	real z_threshold[L];
	real z_ndt[L];
  real z_utility[L];
}

transformed parameters {
  vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
  vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] threshold_t;				// trial-by-trial threshold
	vector<lower=0> [N] ndt_t;							// trial-by-trial ndt

	vector[K] Q;									// Q state values

	real Q_mean;									// mean across all options

	real PE_cor;									// predicion error correct option
	real PE_inc;									// predicion error incorrect option

  real<lower=0, upper=1> alpha_pos_sbj[L];
	real<lower=0, upper=1> alpha_neg_sbj[L];
	real<lower=0> drift_scaling_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];
  real<lower=0> utility_sbj[L];

  real transf_mu_alpha_pos;
	real transf_mu_alpha_neg;
	real transf_mu_drift_scaling;
	real transf_mu_threshold;
	real transf_mu_ndt;
  real transf_mu_utility;

  transf_mu_alpha_pos = Phi(mu_alpha_pos);		// for the output
	transf_mu_alpha_neg = Phi(mu_alpha_neg);
	transf_mu_drift_scaling = log(1 + exp(mu_drift_scaling));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));
  transf_mu_utility = log(1 + exp(mu_utility));

	for (l in 1:L) {
    alpha_pos_sbj[l] = Phi(mu_alpha_pos + z_alpha_pos[l]*sd_alpha_pos);
		alpha_neg_sbj[l] = Phi(mu_alpha_neg + z_alpha_neg[l]*sd_alpha_neg);
		drift_scaling_sbj[l] = log(1 + exp(mu_drift_scaling + z_drift_scaling[l]*sd_drift_scaling));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
    utility_sbj[l] = log(1 + exp(mu_utility + z_utility[l] * sd_utility));
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

		PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];


    if (2*Q[cor_option[n]] - Q[inc_option[n]] > 0){
      drift_cor_t[n] = drift_scaling_sbj[participant[n]]* pow(2*Q[cor_option[n]] - Q[inc_option[n]] + 1, utility_sbj[participant[n]]);
    }else{
      drift_cor_t[n] = drift_scaling_sbj[participant[n]] * pow(exp(2*Q[cor_option[n]] - Q[inc_option[n]]), utility_sbj[participant[n]]);
    }
    if (2*Q[inc_option[n]] - Q[cor_option[n]]>0){
      drift_inc_t[n] = drift_scaling_sbj[participant[n]] * pow(2*Q[inc_option[n]] - Q[cor_option[n]] + 1, utility_sbj[participant[n]]);
    }else{
      drift_inc_t[n] = drift_scaling_sbj[participant[n]] * pow(exp(2*Q[inc_option[n]] - Q[cor_option[n]]), utility_sbj[participant[n]]);
    }

		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];

    if (PE_cor >= 0) {
			Q[cor_option[n]] = Q[cor_option[n]] + alpha_pos_sbj[participant[n]]*PE_cor;
		} else {
			Q[cor_option[n]] = Q[cor_option[n]] + alpha_neg_sbj[participant[n]]*PE_cor;
		}
		if (PE_inc >= 0) {
			Q[inc_option[n]] = Q[inc_option[n]] + alpha_pos_sbj[participant[n]]*PE_inc;
		} else {
			Q[inc_option[n]] = Q[inc_option[n]] + alpha_neg_sbj[participant[n]]*PE_inc;
		}
	}
}

model {
  mu_alpha_pos ~ normal(alpha_pos_priors[1], alpha_pos_priors[2]);
	mu_alpha_neg ~ normal(alpha_neg_priors[1], alpha_neg_priors[2]);
	mu_drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
  mu_utility ~ normal(utility_priors[1], utility_priors[2]);

  sd_alpha_pos ~ normal(alpha_pos_priors[3], alpha_pos_priors[4]);
	sd_alpha_neg ~ normal(alpha_neg_priors[3], alpha_neg_priors[4]);
	sd_drift_scaling ~ normal(drift_scaling_priors[3], drift_scaling_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
  sd_utility ~ normal(utility_priors[3], utility_priors[4]);

  z_alpha_pos ~ normal(0, 1);
	z_alpha_neg ~ normal(0, 1);
	z_drift_scaling ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);
  z_utility ~ normal(0, 1);

  RT ~ race(ndt_t, threshold_t, drift_cor_t, drift_inc_t);
}

generated quantities {
	vector[N] log_lik;
	{
	for (n in 1:N){
		log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(ndt_t, n, 1), segment(threshold_t, n, 1), segment(drift_cor_t, n, 1), segment(drift_inc_t, n, 1));
	}
	}
}
