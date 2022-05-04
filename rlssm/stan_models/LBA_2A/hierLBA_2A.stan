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

     real lba_lpdf(matrix RT, vector rel_sp, vector threshold, vector drift_cor, vector drift_inc, vector ndt){

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
               b = threshold[i] + rel_sp[i];
               t = RT[i,1] - ndt[i];
               if(t > 0){
                    cdf = 1;

                    if(RT[i,2] == 1){
                      pdf = lba_pdf(t, b, threshold[i], drift_cor[i], s);
                      cdf = 1-lba_cdf(t, b, threshold[i], drift_inc[i], s);
                    }
                    else{
                      pdf = lba_pdf(t, b, threshold[i], drift_inc[i], s);
                      cdf = 1-lba_cdf(t, b, threshold[i], drift_cor[i], s);
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
  int<lower=1> L;									// number of levels

  int<lower=1, upper=L> participant[N];			// level (participant)

	int<lower=1,upper=2> accuracy[N];				// 1-> correct, 2->incorrect
	real<lower=0> rt[N];							// rt

  vector[4] rel_sp_priors;
	vector[4] threshold_priors;
  vector[4] ndt_priors;
	vector[4] drift_priors;
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	   RT[n, 1] = rt[n];
	   RT[n, 2] = accuracy[n];
	}
}

parameters {
  real mu_rel_sp;
  real mu_threshold;
  real mu_ndt;
  real mu_drift_cor;
  real mu_drift_inc;

  real<lower=0> sd_rel_sp;
	real<lower=0> sd_threshold;
  real<lower=0> sd_ndt;
	real<lower=0> sd_drift_cor;
	real<lower=0> sd_drift_inc;

  real z_rel_sp[L];
  real z_threshold[L];
  real z_ndt[L];
  real z_drift_cor[L];
  real z_drift_inc[L];
}

transformed parameters {
  vector<lower=0> [N] rel_sp_t;				// trial-by-trial
	vector<lower=0> [N] threshold_t;						// trial-by-trial
  vector<lower=0> [N] ndt_t;				 // trial-by-trial ndt
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions

  real<lower=0> rel_sp_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];
  real<lower=0> drift_cor_sbj[L];
	real<lower=0> drift_inc_sbj[L];

  real<lower=0> transf_mu_rel_sp;
  real<lower=0> transf_mu_threshold;
  real<lower=0> transf_mu_ndt;
	real<lower=0> transf_mu_drift_cor;
	real<lower=0> transf_mu_drift_inc;

  transf_mu_rel_sp = log(1 + exp(mu_rel_sp));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));
	transf_mu_drift_cor = log(1 + exp(mu_drift_cor));
	transf_mu_drift_inc = log(1 + exp(mu_drift_inc));

  for (l in 1:L) {
    rel_sp_sbj[l] = log(1 + exp(mu_rel_sp + z_rel_sp[l]*sd_rel_sp));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
    ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
		drift_cor_sbj[l] = log(1 + exp(mu_drift_cor + z_drift_cor[l]*sd_drift_cor));
		drift_inc_sbj[l] = log(1 + exp(mu_drift_inc + z_drift_inc[l]*sd_drift_inc));
	}

	for (n in 1:N) {
    rel_sp_t[n] = rel_sp_sbj[participant[n]];
		threshold_t[n] = threshold_sbj[participant[n]];
    ndt_t[n] = ndt_sbj[participant[n]];
		drift_cor_t[n] = drift_cor_sbj[participant[n]];
		drift_inc_t[n] = drift_inc_sbj[participant[n]];
	}
}

model {
     mu_rel_sp ~ normal(rel_sp_priors[1], rel_sp_priors[2]);
     mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
     mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
     mu_drift_cor ~ normal(drift_priors[1], drift_priors[2]);
   	 mu_drift_inc ~ normal(drift_priors[1], drift_priors[2]);


     sd_rel_sp ~ normal(rel_sp_priors[3], rel_sp_priors[4]);
     sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
     sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
     sd_drift_cor ~ normal(drift_priors[3], drift_priors[4]);
   	 sd_drift_inc ~ normal(drift_priors[3], drift_priors[4]);

     z_rel_sp ~ normal(0, 1);
     z_threshold ~ normal(0, 1);
     z_ndt ~ normal(0, 1);
     z_drift_cor ~ normal(0, 1);
   	 z_drift_inc ~ normal(0, 1);
     RT ~ lba(rel_sp_t, threshold_t, drift_cor_t, drift_inc_t, ndt_t);
}

generated quantities {
    vector[N] log_lik;
  	{
    	for (n in 1:N){
    		log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(rel_sp_t, n, 1), segment(threshold_t, n, 1), segment(drift_cor_t, n, 1), segment(drift_inc_t, n, 1), segment(ndt_t, n, 1));
    	}
  	}
}
