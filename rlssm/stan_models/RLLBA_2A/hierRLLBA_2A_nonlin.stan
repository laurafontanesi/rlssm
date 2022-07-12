functions{

     real lba_pdf(real t, real b, real A, real v, real s){
          //PDF of the LBA model
          real b_A_tv_ts;
          real b_tv_ts;
          real term_1;
          real term_2;
          real term_3;
          real term_4;
          real pdf;

          b_A_tv_ts = (b - A - t*v)/(t*s);
          b_tv_ts = (b - t*v)/(t*s);
          term_1 = v*Phi(b_A_tv_ts);
          term_2 = s*exp(normal_log(b_A_tv_ts,0,1));
          term_3 = v*Phi(b_tv_ts);
          term_4 = s*exp(normal_log(b_tv_ts,0,1));
          pdf = (1/A)*(-term_1 + term_2 + term_3 - term_4);

          return pdf;
     }

     real lba_cdf(real t, real b, real A, real v, real s){
          //CDF of the LBA model

          real b_A_tv;
          real b_tv;
          real ts;
          real term_1;
          real term_2;
          real term_3;
          real term_4;
          real cdf;

          b_A_tv = b - A - t*v;
          b_tv = b - t*v;
          ts = t*s;
          term_1 = b_A_tv/A * Phi(b_A_tv/ts);
          term_2 = b_tv/A   * Phi(b_tv/ts);
          term_3 = ts/A * exp(normal_log(b_A_tv/ts,0,1));
          term_4 = ts/A * exp(normal_log(b_tv/ts,0,1));
          cdf = 1 + term_1 - term_2 + term_3 - term_4;

          return cdf;

     }

     real lba_lpdf(matrix RT, vector k, vector sp_trial_var, vector drift_cor, vector drift_inc, vector ndt, vector s){

          real t;
          real b;
          real cdf;
          real pdf;
          vector[rows(RT)] prob;
          real out;
          real prob_neg;

          for (i in 1:rows(RT)){
               b = sp_trial_var[i] + k[i];
               t = RT[i,1] - ndt[i];
               if(t > 0){
                    cdf = 1;

                    if(RT[i,2] == 1){
                      pdf = lba_pdf(t, b, sp_trial_var[i], drift_cor[i], s[i]);
                      cdf = 1-lba_cdf(t, b, sp_trial_var[i], drift_inc[i], s[i]);
                    }
                    else{
                      pdf = lba_pdf(t, b, sp_trial_var[i], drift_inc[i], s[i]);
                      cdf = 1-lba_cdf(t, b, sp_trial_var[i], drift_cor[i], s[i]);
                    }
                    prob_neg = Phi(-drift_cor[i]/s[i]) * Phi(-drift_inc[i]/s[i]);
                    prob[i] = pdf*cdf;
                    prob[i] = prob[i]/(1-prob_neg);
                    if(prob[i] < 1e-10){
                         prob[i] = 1e-10;
                    }

               }else{
                    prob[i] = 1e-10;
               }
          }
          out = sum(log(prob));
          return out;
     }
}

data {
	int<lower=1> N;									// number of data items
  int<lower=1> L;								// number of participants
  int<lower=1> K;               // number of options

  int<lower=1, upper=L> participant[N];			// level (participant)

  real initial_value;

  int<lower=1> block_label[N];					// block label
  int<lower=1> trial_block[N];					// trial within block

  vector[N] f_cor;								// feedback correct option
	vector[N] f_inc;								// feedback incorrect option

  int<lower=1, upper=K> cor_option[N];			// correct option
	int<lower=1, upper=K> inc_option[N];			// incorrect option

	int<lower=1,upper=2> accuracy[N];				// 1-> correct, 2->incorrect
	int<lower=0, upper=1> feedback_type[N]; // feedback_type = 0 -> full feedback, feedback_type = 1 -> partial feedback

  real<lower=0> rt[N];							// rt

  vector[4] alpha_priors;             // mean and sd of the prior for alpha
  vector[4] ndt_priors;
  vector[4] k_priors;
	vector[4] sp_trial_var_priors;
  vector[4] slop_priors;
  vector[4] drift_asym_priors;        // mean and sd of the prior for asymtot modulation
	vector[4] drift_scaling_priors;			// mean and sd of the prior for scaling
  vector[4] drift_variability_priors;
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
  real mu_alpha;            // learning rate
  real mu_k;
  real mu_ndt;
  real mu_sp_trial_var;
  real mu_slop;
  real mu_drift_asym;
  real mu_drift_scaling;    // scaling
  real mu_drift_variability;

  real<lower=0> sd_alpha;
  real<lower=0> sd_k;
  real<lower=0> sd_ndt;
  real<lower=0> sd_sp_trial_var;
  real<lower=0> sd_slop;
  real<lower=0> sd_drift_asym;
  real<lower=0> sd_drift_scaling;    // scaling
  real<lower=0> sd_drift_variability;

  real z_alpha[L];            // learning rate
  real z_k[L];
  real z_ndt[L];
  real z_sp_trial_var[L];
  real z_slop[L];
  real z_drift_asym[L];
  real z_drift_scaling[L];    // scaling
  real z_drift_variability[L];    // scaling
}

transformed parameters {
  vector<lower=0> [N] ndt_t;         // trial-by-trial ndt
  vector<lower=0> [N] k_t;				    // trial-by-trial
	vector<lower=0> [N] sp_trial_var_t;						// trial-by-trial
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
  vector<lower=0> [N] drift_variability_t;

  real PE_cor;			// prediction error correct option
	real PE_inc;			// prediction error incorrect option
	vector[K] Q;

  real Q_mean;
  real Q_min;
  real Q_mean_pres[N];

  real<lower=0, upper=1> alpha_sbj[L];
  real<lower=0> k_sbj[L];
  real<lower=0> ndt_sbj[L];
	real<lower=0> sp_trial_var_sbj[L];
  real<lower=0> slop_sbj[L];
  real<lower=0> drift_asym_sbj[L];
	real<lower=0> drift_scaling_sbj[L];
  real<lower=0> drift_variability_sbj[L];

  real<lower=0, upper=1> transf_mu_alpha;
  real<lower=0> transf_mu_k;
  real<lower=0> transf_mu_ndt;
  real<lower=0> transf_mu_sp_trial_var;
  real<lower=0> transf_mu_slop;
  real<lower=0> transf_mu_drift_asym;
	real<lower=0> transf_mu_drift_scaling;
  real<lower=0> transf_mu_drift_variability;

  transf_mu_alpha = Phi(mu_alpha);
  transf_mu_k = log(1 + exp(mu_k));
  transf_mu_ndt = log(1 + exp(mu_ndt));
	transf_mu_sp_trial_var = log(1 + exp(mu_sp_trial_var));  
  transf_mu_slop = log(1 + exp(mu_slop));
  transf_mu_drift_asym = log(1 + exp(mu_drift_asym));
	transf_mu_drift_scaling = log(1 + exp(mu_drift_scaling));
  transf_mu_drift_variability = log(1 + exp(mu_drift_variability));

  for (l in 1:L) {
    alpha_sbj[l] = Phi(mu_alpha + z_alpha[l]*sd_alpha);
    k_sbj[l] = log(1 + exp(mu_k + z_k[l]*sd_k));
    ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
    sp_trial_var_sbj[l] = log(1 + exp(mu_sp_trial_var + z_sp_trial_var[l]*sd_sp_trial_var));
    slop_sbj[l] = log(1 + exp(mu_slop + z_slop[l] * sd_slop));
    drift_asym_sbj[l] = log(1 + exp(mu_drift_asym + z_drift_asym[l]*sd_drift_asym));
		drift_scaling_sbj[l] = log(1 + exp(mu_drift_scaling + z_drift_scaling[l]*sd_drift_scaling));
    drift_variability_sbj[l] = log(1 + exp(mu_drift_variability + z_drift_variability[l]*sd_drift_variability));
	}

	for (n in 1:N) {
    if (trial_block[n] == 1){
			if (block_label[n] == 1){
				Q = Q0;
			} else{
				Q_mean = mean(Q);
				Q = rep_vector(Q_mean, K);
			}
		}
    Q_min = min(Q);
    Q_mean_pres[n] = (Q[cor_option[n]] + Q[inc_option[n]])/2;


    PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];

    drift_cor_t[n] = (drift_scaling_sbj[participant[n]]+0.1*drift_asym_sbj[participant[n]]*(Q_mean_pres[n] - Q_min)) / ( 1+exp(slop_sbj[participant[n]]*(Q_mean_pres[n] - Q[cor_option[n]])) );
    drift_inc_t[n] = (drift_scaling_sbj[participant[n]]+0.1*drift_asym_sbj[participant[n]]*(Q_mean_pres[n] - Q_min)) / ( 1+exp(slop_sbj[participant[n]]*(Q_mean_pres[n] - Q[inc_option[n]])) );

    k_t[n] = k_sbj[participant[n]];
    sp_trial_var_t[n] = sp_trial_var_sbj[participant[n]];
    ndt_t[n] = ndt_sbj[participant[n]];
    drift_variability_t[n] = drift_variability_sbj[participant[n]];

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
  mu_k ~ normal(k_priors[1], k_priors[2]);
  mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
  mu_sp_trial_var ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
  mu_slop ~ normal(slop_priors[1], slop_priors[2]);
  mu_drift_asym ~ normal(drift_asym_priors[1], drift_asym_priors[2]);
  mu_drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
  mu_drift_variability ~ normal(drift_variability_priors[1], drift_variability_priors[2]);

  sd_alpha ~ normal(alpha_priors[3], alpha_priors[4]);
  sd_k ~ normal(k_priors[3], k_priors[4]);
  sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
  sd_sp_trial_var ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
  sd_slop ~ normal(slop_priors[3], slop_priors[4]);
  sd_drift_asym ~ normal(drift_asym_priors[3], drift_asym_priors[4]);
  sd_drift_scaling ~ normal(drift_scaling_priors[3], drift_scaling_priors[4]);
  sd_drift_variability ~ normal(drift_variability_priors[3], drift_variability_priors[4]);

  z_alpha ~ normal(0, 1);
  z_k ~ normal(0, 1);
  z_ndt ~ normal(0, 1);
  z_sp_trial_var ~ normal(0, 1);
  z_slop ~ normal(0, 1);
  z_drift_asym ~ normal(0, 1);   
  z_drift_scaling ~ normal(0, 1);
  z_drift_variability ~ normal(0, 1);

  RT ~ lba(k_t, sp_trial_var_t, drift_cor_t, drift_inc_t, ndt_t, drift_variability_t);
}

generated quantities {
    vector[N] log_lik;
    {
      for (n in 1:N){
        log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(k_t, n, 1), segment(sp_trial_var_t, n, 1), segment(drift_cor_t, n, 1), segment(drift_inc_t, n, 1), segment(ndt_t, n, 1), segment(drift_variability_t, n, 1));
      }
    }
}