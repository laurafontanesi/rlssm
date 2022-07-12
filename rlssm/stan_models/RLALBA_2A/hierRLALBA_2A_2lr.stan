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

     real lba_lpdf(matrix RT, vector k, vector sp_trial_var, vector drift_cor, vector drift_inc, vector ndt){

          real t;
          real b;
          real cdf;
          real pdf;
          vector[rows(RT)] prob;
          real out;
          real prob_neg;
          real s;
          s = 1;

          for (i in 1:rows(RT)){
               b = sp_trial_var[i] + k[i];
               t = RT[i,1] - ndt[i];
               if(t > 0){
                    cdf = 1;

                    if(RT[i,2] == 1){
                      pdf = lba_pdf(t, b, sp_trial_var[i], drift_cor[i], s);
                      cdf = 1-lba_cdf(t, b, sp_trial_var[i], drift_inc[i], s);
                    }
                    else{
                      pdf = lba_pdf(t, b, sp_trial_var[i], drift_inc[i], s);
                      cdf = 1-lba_cdf(t, b, sp_trial_var[i], drift_cor[i], s);
                    }
                    prob_neg = Phi(-drift_cor[i]/s) * Phi(-drift_inc[i]/s);
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
	real<lower=0> rt[N];							// rt
     int<lower=0, upper=1> feedback_type[N]; // feedback_type = 0 -> full feedback, feedback_type = 1 -> partial feedback

  vector[4] k_priors;
	vector[4] sp_trial_var_priors;
  vector[4] ndt_priors;
  vector[4] v0_priors;
  vector[4] ws_priors;
  vector[4] wd_priors;
  vector[4] alpha_pos_priors;						// mean and sd of the mu_alpha_pos prior and sd_alpha_pos prior
	vector[4] alpha_neg_priors;						// mean and sd of the mu_alpha_neg prior and sd_alpha_neg prior
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
   real mu_k;
   real mu_sp_trial_var;
   real mu_ndt;
   real mu_v0;
   real mu_ws;
   real mu_wd;
   real mu_alpha_pos;
 	 real mu_alpha_neg;

   real<lower=0> sd_k;
   real<lower=0> sd_sp_trial_var;
   real<lower=0> sd_ndt;
   real<lower=0> sd_v0;
   real<lower=0> sd_ws;
   real<lower=0> sd_wd;
   real<lower=0> sd_alpha_pos;
 	 real<lower=0> sd_alpha_neg;

   real z_k[L];
   real z_sp_trial_var[L];
   real z_ndt[L];
   real z_v0[L];
   real z_ws[L];
   real z_wd[L];
   real z_alpha_pos[L];
 	 real z_alpha_neg[L];
}

transformed parameters {
  vector<lower=0> [N] k_t;				    // trial-by-trial
	vector<lower=0> [N] sp_trial_var_t;						// trial-by-trial
  vector<lower=0> [N] ndt_t;				 // trial-by-trial ndt
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions

  real PE_cor;			// prediction error correct option
	real PE_inc;			// prediction error incorrect option
	vector[K] Q;

  real Q_mean;

  real<lower=0> k_sbj[L];
	real<lower=0> sp_trial_var_sbj[L];
  real<lower=0> ndt_sbj[L];
  real<lower=0> v0_sbj[L];
  real<lower=0> ws_sbj[L];
  real<lower=0> wd_sbj[L];
  real<lower=0, upper=1> alpha_pos_sbj[L];
	real<lower=0, upper=1> alpha_neg_sbj[L];

  real<lower=0> transf_mu_k;
  real<lower=0> transf_mu_sp_trial_var;
  real<lower=0> transf_mu_ndt;
  real<lower=0> transf_mu_v0;
  real<lower=0> transf_mu_ws;
  real<lower=0> transf_mu_wd;
  real<lower=0, upper=1> transf_mu_alpha_pos;
	real<lower=0, upper=1> transf_mu_alpha_neg;

  transf_mu_k = log(1 + exp(mu_k)); // for the output
	transf_mu_sp_trial_var = log(1 + exp(mu_sp_trial_var));
	transf_mu_ndt = log(1 + exp(mu_ndt));
  transf_mu_v0 = log(1 + exp(mu_v0));
  transf_mu_ws = log(1 + exp(mu_ws));
  transf_mu_wd = log(1 + exp(mu_wd));
  transf_mu_alpha_pos = Phi(mu_alpha_pos);
	transf_mu_alpha_neg = Phi(mu_alpha_neg);

  for (l in 1:L) {
    k_sbj[l] = log(1 + exp(mu_k + z_k[l]*sd_k));
    sp_trial_var_sbj[l] = log(1 + exp(mu_sp_trial_var + z_sp_trial_var[l]*sd_sp_trial_var));
    ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
    v0_sbj[l] = log(1 + exp(mu_v0 + z_v0[l]*sd_v0));
    ws_sbj[l] = log(1 + exp(mu_ws + z_ws[l]*sd_ws));
    wd_sbj[l] = log(1 + exp(mu_wd + z_wd[l]*sd_wd));
    alpha_pos_sbj[l] = Phi(mu_alpha_pos + z_alpha_pos[l]*sd_alpha_pos);
		alpha_neg_sbj[l] = Phi(mu_alpha_neg + z_alpha_neg[l]*sd_alpha_neg);
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

    PE_cor = f_cor[n] - Q[cor_option[n]];
		PE_inc = f_inc[n] - Q[inc_option[n]];

    k_t[n] = k_sbj[participant[n]];
		sp_trial_var_t[n] = sp_trial_var_sbj[participant[n]];
    ndt_t[n] = ndt_sbj[participant[n]];
    drift_cor_t[n] = v0_sbj[participant[n]] + wd_sbj[participant[n]]*(Q[cor_option[n]]-Q[inc_option[n]]) + ws_sbj[participant[n]]*(Q[cor_option[n]]+Q[inc_option[n]]);
    drift_inc_t[n] = v0_sbj[participant[n]] + wd_sbj[participant[n]]*(Q[inc_option[n]]-Q[cor_option[n]]) + ws_sbj[participant[n]]*(Q[cor_option[n]]+Q[inc_option[n]]);

    if (feedback_type[n] == 1){
      if(accuracy[n] == 1){
        if (PE_cor >= 0) {
          Q[cor_option[n]] = Q[cor_option[n]] + alpha_pos_sbj[participant[n]]*PE_cor;
        } else {
          Q[cor_option[n]] = Q[cor_option[n]] + alpha_neg_sbj[participant[n]]*PE_cor;
        }
      }
      else{
        if (PE_inc >= 0) {
          Q[inc_option[n]] = Q[inc_option[n]] + alpha_pos_sbj[participant[n]]*PE_inc;
        } else {
          Q[inc_option[n]] = Q[inc_option[n]] + alpha_neg_sbj[participant[n]]*PE_inc;
        }
      }
    }
    else{
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
}

model {
     mu_k ~ normal(k_priors[1], k_priors[2]);
     mu_sp_trial_var ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
     mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
     mu_v0 ~ normal(v0_priors[1], v0_priors[2]);
     mu_ws ~ normal(ws_priors[1], ws_priors[2]);
     mu_wd ~ normal(wd_priors[1], wd_priors[2]);
     mu_alpha_pos ~ normal(alpha_pos_priors[1], alpha_pos_priors[2]);
   	 mu_alpha_neg ~ normal(alpha_neg_priors[1], alpha_neg_priors[2]);

     sd_k ~ normal(k_priors[3], k_priors[4]);
     sd_sp_trial_var ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
     sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
     sd_v0 ~ normal(v0_priors[3], v0_priors[4]);
     sd_ws ~ normal(ws_priors[3], ws_priors[4]);
     sd_wd ~ normal(wd_priors[3], wd_priors[4]);
     sd_alpha_pos ~ normal(alpha_pos_priors[3], alpha_pos_priors[4]);
   	 sd_alpha_neg ~ normal(alpha_neg_priors[3], alpha_neg_priors[4]);

     z_k ~ normal(0, 1);
     z_sp_trial_var ~ normal(0, 1);
     z_ndt ~ normal(0, 1);
     z_v0 ~ normal(0, 1);
     z_ws ~ normal(0, 1);
     z_wd ~ normal(0, 1);
     z_alpha_pos ~ normal(0, 1);
   	 z_alpha_neg ~ normal(0, 1);

     RT ~ lba(k_t, sp_trial_var_t, drift_cor_t, drift_inc_t, ndt_t);
}

generated quantities {
    vector[N] log_lik;
  	{
    	for (n in 1:N){
    		log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(k_t, n, 1), segment(sp_trial_var_t, n, 1), segment(drift_cor_t, n, 1), segment(drift_inc_t, n, 1), segment(ndt_t, n, 1));
    	}
  	}
}
