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

     real lba_lpdf(matrix RT, vector k, vector A, vector drift_cor, vector drift_inc, vector tau){

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
               b = A[i] + k[i];
               t = RT[i,1] - tau[i];
               if(t > 0){
                    cdf = 1;

                    if(RT[i,2] == 1){
                      pdf = lba_pdf(t, b, A[i], drift_cor[i], s);
                      cdf = 1-lba_cdf(t, b, A[i], drift_inc[i], s);
                    }
                    else{
                      pdf = lba_pdf(t, b, A[i], drift_inc[i], s);
                      cdf = 1-lba_cdf(t, b, A[i], drift_cor[i], s);
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

  vector[N] S_cor;								// subjective perception of correct option
	vector[N] S_inc;								// subjective perception of incorrect option

  vector[4] k_priors;
	vector[4] A_priors;
  vector[4] tau_priors;
  vector[4] v0_priors;
  vector[4] ws_priors;
  vector[4] wd_priors;
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	   RT[n, 1] = rt[n];
	   RT[n, 2] = accuracy[n];
	}
}

parameters {
  real mu_k;
  real mu_A;
  real mu_tau;
  real mu_v0;
  real mu_ws;
  real mu_wd;

  real<lower=0> sd_k;
	real<lower=0> sd_A;
  real<lower=0> sd_tau;
	real<lower=0> sd_v0;
	real<lower=0> sd_ws;
  real<lower=0> sd_wd;

  real z_k[L];
  real z_A[L];
  real z_tau[L];
  real z_v0[L];
  real z_ws[L];
  real z_wd[L];
}

transformed parameters {
  vector<lower=0> [N] k_t;				// trial-by-trial
	vector<lower=0> [N] A_t;						// trial-by-trial
  vector<lower=0> [N] tau_t;				 // trial-by-trial ndt
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions

  real<lower=0> k_sbj[L];
	real<lower=0> A_sbj[L];
	real<lower=0> tau_sbj[L];
  real<lower=0> v0_sbj[L];
	real<lower=0> ws_sbj[L];
  real<lower=0> wd_sbj[L];

  real<lower=0> transf_mu_k;
  real<lower=0> transf_mu_A;
  real<lower=0> transf_mu_tau;
	real<lower=0> transf_mu_v0;
	real<lower=0> transf_mu_ws;
  real<lower=0> transf_mu_wd;

  transf_mu_k = log(1 + exp(mu_k));
	transf_mu_A = log(1 + exp(mu_A));
	transf_mu_tau = log(1 + exp(mu_tau));
	transf_mu_v0 = log(1 + exp(mu_v0));
	transf_mu_ws = log(1 + exp(mu_ws));
  transf_mu_wd = log(1 + exp(mu_wd));

  for (l in 1:L) {
    k_sbj[l] = log(1 + exp(mu_k + z_k[l]*sd_k));
		A_sbj[l] = log(1 + exp(mu_A + z_A[l]*sd_A));
    tau_sbj[l] = log(1 + exp(mu_tau + z_tau[l]*sd_tau));
		v0_sbj[l] = log(1 + exp(mu_v0 + z_v0[l]*sd_v0));
		ws_sbj[l] = log(1 + exp(mu_ws + z_ws[l]*sd_ws));
    wd_sbj[l] = log(1 + exp(mu_wd + z_wd[l]*sd_wd));
	}

	for (n in 1:N) {
    k_t[n] = k_sbj[participant[n]];
		A_t[n] = A_sbj[participant[n]];
    tau_t[n] = tau_sbj[participant[n]];
		drift_cor_t[n] = v0_sbj[participant[n]] + wd_sbj[participant[n]]*(S_cor[n]-S_inc[n]) + ws_sbj[participant[n]]*(S_cor[n]+S_inc[n]);
		drift_inc_t[n] = v0_sbj[participant[n]] + wd_sbj[participant[n]]*(S_inc[n]-S_cor[n]) + ws_sbj[participant[n]]*(S_cor[n]+S_inc[n]);
	}
}

model {
     mu_k ~ normal(k_priors[1], k_priors[2]);
     mu_A ~ normal(A_priors[1], A_priors[2]);
     mu_tau ~ normal(tau_priors[1], tau_priors[2]);
     mu_v0 ~ normal(v0_priors[1], v0_priors[2]);
   	 mu_ws ~ normal(ws_priors[1], ws_priors[2]);
     mu_wd ~ normal(wd_priors[1], wd_priors[2]);

     sd_k ~ normal(k_priors[3], k_priors[4]);
     sd_A ~ normal(A_priors[3], A_priors[4]);
     sd_tau ~ normal(tau_priors[3], tau_priors[4]);
     sd_v0 ~ normal(v0_priors[3], v0_priors[4]);
   	 sd_ws ~ normal(ws_priors[3], ws_priors[4]);
     sd_wd ~ normal(wd_priors[3], wd_priors[4]);

     z_k ~ normal(0, 1);
     z_A ~ normal(0, 1);
     z_tau ~ normal(0, 1);
     z_v0 ~ normal(0, 1);
   	 z_ws ~ normal(0, 1);
     z_wd ~ normal(0, 1);

     RT ~ lba(k_t, A_t, drift_cor_t, drift_inc_t, tau_t);
}

generated quantities {
    vector[N] log_lik;
  	{
    	for (n in 1:N){
    		log_lik[n] = lba_lpdf(block(RT, n, 1, 1, 2)| segment(k_t, n, 1), segment(A_t, n, 1), segment(drift_cor_t, n, 1), segment(drift_inc_t, n, 1), segment(tau_t, n, 1));
    	}
  	}
}
