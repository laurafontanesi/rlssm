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

     real race_lpdf(matrix RT, vector ndt, vector b, vector drift_cor, vector drift_inc){

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

data {
	int<lower=1> N;									// number of data items
	int<lower=1> L;									// number of levels
	int<lower=1, upper=L> participant[N];			// level (participant)

	int<lower=1,upper=2> accuracy[N];				// 1-> correct, 2->incorrect
	real<lower=0> rt[N];							// rt

	vector[4] drift_priors;							// mean and sd of the group mean and of the group sd hyper-priors
	vector[4] threshold_priors;						// mean and sd of the group mean and of the group sd hyper-priors
	vector[4] ndt_priors;							// mean and sd of the group mean and of the group sd hyper-priors
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	RT[n, 1] = rt[n];
	RT[n, 2] = accuracy[n];
	}
}

parameters {
	real mu_ndt;
	real mu_threshold;
	real mu_drift_cor;
	real mu_drift_inc;

	real<lower=0> sd_ndt;
	real<lower=0> sd_threshold;
	real<lower=0> sd_drift_cor;
	real<lower=0> sd_drift_inc;

	real z_ndt[L];
	real z_threshold[L];
	real z_drift_cor[L];
	real z_drift_inc[L];
}

transformed parameters {
	vector<lower=0>[N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0>[N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0>[N] threshold_t;				// trial-by-trial threshold
	vector<lower=0>[N] ndt_t;						// trial-by-trial ndt

	real<lower=0> drift_cor_sbj[L];
	real<lower=0> drift_inc_sbj[L];
	real<lower=0> threshold_sbj[L];
	real<lower=0> ndt_sbj[L];

	real<lower=0> transf_mu_drift_cor;
	real<lower=0> transf_mu_drift_inc;
	real<lower=0> transf_mu_threshold;
	real<lower=0> transf_mu_ndt;

	transf_mu_drift_cor = log(1 + exp(mu_drift_cor));
	transf_mu_drift_inc = log(1 + exp(mu_drift_inc));
	transf_mu_threshold = log(1 + exp(mu_threshold));
	transf_mu_ndt = log(1 + exp(mu_ndt));

	for (l in 1:L) {
		drift_cor_sbj[l] = log(1 + exp(mu_drift_cor + z_drift_cor[l]*sd_drift_cor));
		drift_inc_sbj[l] = log(1 + exp(mu_drift_inc + z_drift_inc[l]*sd_drift_inc));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
		ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
	}

	for (n in 1:N) {
		drift_cor_t[n] = drift_cor_sbj[participant[n]];
		drift_inc_t[n] = drift_inc_sbj[participant[n]];
		threshold_t[n] = threshold_sbj[participant[n]];
		ndt_t[n] = ndt_sbj[participant[n]];
	}
}

model {
	mu_drift_cor ~ normal(drift_priors[1], drift_priors[2]);
	mu_drift_inc ~ normal(drift_priors[1], drift_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);

	sd_drift_cor ~ normal(drift_priors[3], drift_priors[4]);
	sd_drift_inc ~ normal(drift_priors[3], drift_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
	sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);

	z_drift_cor ~ normal(0, 1);
	z_drift_inc ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
	z_ndt ~ normal(0, 1);

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
