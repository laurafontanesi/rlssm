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
                      cdf = 1-lba_cdf(t| b, sp_trial_var[i], drift_inc[i], s[i]);
                    }
                    else{
                      pdf = lba_pdf(t, b, sp_trial_var[i], drift_inc[i], s[i]);
                      cdf = 1-lba_cdf(t| b, sp_trial_var[i], drift_cor[i], s[i]);
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

	array[N] int<lower=1,upper=2> accuracy;				// 1-> correct, 2->incorrect
	array[N] real<lower=0> rt;							// rt

     vector[2] k_priors;
	vector[2] sp_trial_var_priors;
     vector[2] ndt_priors;
	vector[2] drift_priors;
     vector[2] drift_variability_priors;
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	   RT[n, 1] = rt[n];
	   RT[n, 2] = accuracy[n];
	}
}

parameters {
   real k;
   real sp_trial_var;
   real ndt;
   real drift_cor;
   real drift_inc;
   real drift_variability;
}


transformed parameters {
     vector<lower=0> [N] k_t;				// trial-by-trial
	vector<lower=0> [N] sp_trial_var_t;						// trial-by-trial
     vector<lower=0> [N] ndt_t;				 // trial-by-trial ndt
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
     vector<lower=0> [N] drift_variability_t;

     real<lower=0> transf_k;
     real<lower=0> transf_sp_trial_var;
     real<lower=0> transf_ndt;
	real<lower=0> transf_drift_cor;
	real<lower=0> transf_drift_inc;
     real<lower=0> transf_drift_variability;

     transf_k = log(1 + exp(k));
	transf_sp_trial_var = log(1 + exp(sp_trial_var));
	transf_ndt = log(1 + exp(ndt));
	transf_drift_cor = log(1 + exp(drift_cor));
	transf_drift_inc = log(1 + exp(drift_inc));
     transf_drift_variability = log(1 + exp(drift_variability));

	for (n in 1:N) {
          k_t[n] = transf_k;
		sp_trial_var_t[n] = transf_sp_trial_var;
          ndt_t[n] = transf_ndt;
		drift_cor_t[n] = transf_drift_cor;
		drift_inc_t[n] = transf_drift_inc;
          drift_variability_t[n] = transf_drift_variability;
	}
}

model {
     k ~ normal(k_priors[1], k_priors[2]);
     sp_trial_var ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
     ndt ~ normal(ndt_priors[1], ndt_priors[2]);
     drift_cor ~ normal(drift_priors[1], drift_priors[2]);
   	drift_inc ~ normal(drift_priors[1], drift_priors[2]);
     drift_variability ~ normal(drift_variability_priors[1], drift_variability_priors[2]);

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
