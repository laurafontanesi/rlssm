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

  vector[4] threshold_priors;					// mean and sd of the prior for threshold
	vector[4] ndt_priors;							  // mean and sd of the prior for non-decision time
  vector[4] v0_priors;
  vector[4] ws_priors;
  vector[4] wd_priors;
	vector[4] alpha_priors;             // mean and sd of the prior for alpha
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
	real mu_threshold;
	real mu_ndt;
  real mu_v0;
  real mu_ws;
  real mu_wd;
  real mu_alpha;

	real<lower=0> sd_threshold;
	real<lower=0> sd_ndt;
  real<lower=0> sd_v0;
  real<lower=0> sd_ws;
  real<lower=0> sd_wd;
  real<lower=0> sd_alpha;

	real z_threshold[L];
	real z_ndt[L];
  real z_v0[L];
  real z_ws[L];
  real z_wd[L];
  real z_alpha[L];
}

transformed parameters {
  vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
  vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] threshold_t;				// trial-by-trial threshold
	vector<lower=0> [N] ndt_t;							// trial-by-trial ndt

	vector[K] Q;									// Q state values

	real Q_mean;									// mean across all options
	real Q_mean_pres[N];							// mean Q presented options
	real PE_cor;									// predicion error correct option
	real PE_inc;									// predicion error incorrect option

	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];
  real<lower=0> v0_sbj[L];
  real<lower=0> ws_sbj[L];
  real<lower=0> wd_sbj[L];
  real<lower=0, upper=1> alpha_sbj[L];

	real<lower=0> transf_mu_threshold;
	real<lower=0> transf_mu_ndt;
  real<lower=0> transf_mu_v0;
  real<lower=0> transf_mu_ws;
  real<lower=0> transf_mu_wd;
  real<lower=0, upper=1> transf_mu_alpha;

	transf_mu_threshold = log(1 + exp(mu_threshold)); // for the output
	transf_mu_ndt = log(1 + exp(mu_ndt));
  transf_mu_v0 = log(1 + exp(mu_v0));
  transf_mu_ws = log(1 + exp(mu_ws));
  transf_mu_wd = log(1 + exp(mu_wd));
  transf_mu_alpha = Phi(mu_alpha);

	for (l in 1:L) {
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
    v0_sbj[l] = log(1 + exp(mu_v0 + z_v0[l]*sd_v0));
    ws_sbj[l] = log(1 + exp(mu_ws + z_ws[l]*sd_ws));
    wd_sbj[l] = log(1 + exp(mu_wd + z_wd[l]*sd_wd));
    alpha_sbj[l] = Phi(mu_alpha + z_alpha[l]*sd_alpha);
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

    threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];

    drift_cor_t[n] = v0_sbj[participant[n]] + wd_sbj[participant[n]]*(Q[cor_option[n]]-Q[inc_option[n]]) + ws_sbj[participant[n]]*(Q[cor_option[n]]+Q[inc_option[n]]);
    drift_inc_t[n] = v0_sbj[participant[n]] + wd_sbj[participant[n]]*(Q[inc_option[n]]-Q[cor_option[n]]) + ws_sbj[participant[n]]*(Q[cor_option[n]]+Q[inc_option[n]]);

		Q[cor_option[n]] = Q[cor_option[n]] + alpha_sbj[participant[n]]*PE_cor;
		Q[inc_option[n]] = Q[inc_option[n]] + alpha_sbj[participant[n]]*PE_inc;
	}
}

model {
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
  mu_v0 ~ normal(v0_priors[1], v0_priors[2]);
  mu_ws ~ normal(ws_priors[1], ws_priors[2]);
  mu_wd ~ normal(wd_priors[1], wd_priors[2]);
  mu_alpha ~ normal(alpha_priors[1], alpha_priors[2]);

	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
  sd_v0 ~ normal(v0_priors[3], v0_priors[4]);
  sd_ws ~ normal(ws_priors[3], ws_priors[4]);
  sd_wd ~ normal(wd_priors[3], wd_priors[4]);
  sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);

	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);
  z_v0 ~ normal(0, 1);
  z_ws ~ normal(0, 1);
  z_wd ~ normal(0, 1);
  z_alpha ~ normal(0, 1);

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
