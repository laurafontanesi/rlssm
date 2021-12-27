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

	int<lower=1,upper=2> accuracy[N];				// 1-> correct, 2->incorrect
	real<lower=0> rt[N];							// rt

  vector[N] S_cor;								// subjective perception of correct option
	vector[N] S_inc;								// subjective perception of incorrect option

  vector[2] k_priors;
	vector[2] A_priors;
  vector[2] tau_priors;
	vector[2] v0_priors;
  vector[2] ws_priors;
  vector[2] wd_priors;
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
   real A;
   real tau;
   real v0;
   real ws;
   real wd;
}


transformed parameters {
  vector<lower=0> [N] k_t;				// trial-by-trial
	vector<lower=0> [N] A_t;						// trial-by-trial
  vector<lower=0> [N] tau_t;				 // trial-by-trial ndt
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions

  real<lower=0> transf_k;
  real<lower=0> transf_A;
  real<lower=0> transf_tau;
	real<lower=0> transf_v0;
	real<lower=0> transf_ws;
  real<lower=0> transf_wd;

  transf_k = log(1 + exp(k));
	transf_A = log(1 + exp(A));
	transf_tau = log(1 + exp(tau));
	transf_v0 = log(1 + exp(v0));
	transf_ws = log(1 + exp(ws));
  transf_wd = log(1 + exp(wd));

	for (n in 1:N) {
    k_t[n] = transf_k;
		A_t[n] = transf_A;
    tau_t[n] = transf_tau;
		drift_cor_t[n] = transf_v0 + transf_wd * (S_cor[n] - S_inc[n]) + transf_ws * (S_cor[n]+S_inc[n]);
		drift_inc_t[n] = transf_v0 + transf_wd * (S_inc[n] - S_cor[n]) + transf_ws * (S_cor[n]+S_inc[n]);
	}
}

model {
     k ~ normal(k_priors[1], k_priors[2]);
     A ~ normal(A_priors[1], A_priors[2]);
     tau ~ normal(tau_priors[1], tau_priors[2]);
     v0 ~ normal(v0_priors[1], v0_priors[2]);
     ws ~ normal(ws_priors[1], ws_priors[2]);
     wd ~ normal(wd_priors[1], wd_priors[2]);

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
