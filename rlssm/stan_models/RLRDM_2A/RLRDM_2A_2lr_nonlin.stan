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
  int<lower=1> K;               // number of options
  real initial_value;
  int<lower=1> block_label[N];					// block label
  int<lower=1> trial_block[N];					// trial within block

  vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option


  int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option
  int<lower=1, upper=2> accuracy[N];				// accuracy (1->cor, 2->inc)

  real<lower=0> rt[N];							// reaction time

  vector[2] alpha_pos_priors;						// mean and sd of the alpha_pos prior
	vector[2] alpha_neg_priors;						// mean and sd of the alpha_neg prior
  vector[2] utility_priors;
	vector[2] drift_scaling_priors;			// mean and sd of the prior for scaling
	vector[2] threshold_priors;					// mean and sd of the prior for threshold
	vector[2] ndt_priors;							  // mean and sd of the prior for non-decision time
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
	real ndt;     // non-decision time
  real threshold;       // threshold
  real drift_scaling;    // scaling
  real alpha_pos;
	real alpha_neg;
  real utility;
  real theta;
}

transformed parameters {
  vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] threshold_t;				// trial-by-trial threshold
	vector<lower=0> [N] ndt_t;						// trial-by-trial ndt

  real PE_cor;			// prediction error correct option
	real PE_inc;			// prediction error incorrect option
	vector[K] Q;

  real Q_mean;
  // real delta_Q_cor[N];
  // real delta_Q_inc[N];
  // real Q_mean_pres[N];							// mean Q presented options


  real transf_alpha_pos;
	real transf_alpha_neg;
  real transf_ndt;
	real transf_drift_scaling;
  real transf_threshold;
  real transf_utility;
  real transf_theta;

	transf_drift_scaling = log(1 + exp(drift_scaling)); // for the output
  transf_threshold = log(1 + exp(threshold));
  transf_ndt = log(1 + exp(ndt));
  transf_alpha_pos = Phi(alpha_pos);
	transf_alpha_neg = Phi(alpha_neg);
  transf_utility = log(1 + exp(utility));
  transf_theta = log(1 + exp(theta));



  for (n in 1:N){
    if (trial_block[n] == 1){
			if (block_label[n] == 1){
				Q = Q0;
			} else{
				Q_mean = mean(Q);
				Q = rep_vector(Q_mean, K);
			}
		}

    // Q_mean_pres[n] = (Q[cor_option[n]] + Q[inc_option[n]])/2;
    // delta_Q_cor[n] = Q[cor_option[n]] - Q[inc_option[n]];
    // delta_Q_inc[n] = Q[inc_option[n]] - Q[cor_option[n]];

    PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];

    // drift_cor_t[n] = transf_drift_scaling * pow(Q[cor_option[n]], transf_utility);
    // drift_inc_t[n] = transf_drift_scaling * pow(Q[inc_option[n]], transf_utility);

    // drift_cor_t[n] = transf_drift_scaling * pow((2*Q[cor_option[n]] - Q[inc_option[n]])/2, transf_utility);
    // drift_inc_t[n] = transf_drift_scaling * pow((2*Q[inc_option[n]] - Q[cor_option[n]])/2, transf_utility);

    // drift_cor_t[n] = transf_drift_scaling * pow(Q_mean_pres[n] + delta_Q_cor[n], transf_utility);
    // drift_inc_t[n] = transf_drift_scaling * pow(Q_mean_pres[n] + delta_Q_inc[n], transf_utility);

    if (2*Q[cor_option[n]] - Q[inc_option[n]] > 0){
      drift_cor_t[n] = transf_drift_scaling * pow(2*Q[cor_option[n]] - Q[inc_option[n]] + transf_theta, transf_utility);
    }else{
      drift_cor_t[n] = transf_drift_scaling * pow(transf_theta * exp((2*Q[cor_option[n]] - Q[inc_option[n]])/transf_theta), transf_utility);
    }
    if (2*Q[inc_option[n]] - Q[cor_option[n]]>0){
      drift_inc_t[n] = transf_drift_scaling * pow(2*Q[inc_option[n]] - Q[cor_option[n]] + transf_theta, transf_utility);
    }else{
      drift_inc_t[n] = transf_drift_scaling * pow(transf_theta * exp((2*Q[inc_option[n]] - Q[cor_option[n]])/transf_theta), transf_utility);
    }

    threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;

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

model {
	ndt ~  normal(ndt_priors[1], ndt_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
  alpha_pos ~ normal(alpha_pos_priors[1], alpha_pos_priors[2]);
	alpha_neg ~ normal(alpha_neg_priors[1], alpha_neg_priors[2]);
	drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
  utility ~ normal(utility_priors[1], utility_priors[2]);
  theta ~ normal(1, 1);

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
