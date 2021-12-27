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

data {
	int<lower=1> N;									// number of data items

	int<lower=1,upper=2> accuracy[N];				// 1-> correct, 2->incorrect
	real<lower=0> rt[N];							// rt

	vector[2] drift_priors;							// mean and sd of the prior
	vector[2] threshold_priors;						// mean and sd of the prior
	vector[2] ndt_priors;							// mean and sd of the prior
}

transformed data {
	matrix [N, 2] RT;

	for (n in 1:N){
	RT[n, 1] = rt[n];
	RT[n, 2] = accuracy[n];
	}
}

parameters {
	real ndt;
	real threshold;
	real drift_cor;
	real drift_inc;
}

transformed parameters {
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] threshold_t;				// trial-by-trial threshold
	vector<lower=0> [N] ndt_t;						// trial-by-trial ndt

	real<lower=0> transf_drift_cor;
	real<lower=0> transf_drift_inc;
	real<lower=0> transf_threshold;
	real<lower=0> transf_ndt;

	transf_drift_cor = log(1 + exp(drift_cor));
	transf_drift_inc = log(1 + exp(drift_inc));
	transf_threshold = log(1 + exp(threshold));
	transf_ndt = log(1 + exp(ndt));

	for (n in 1:N) {
		drift_cor_t[n] = transf_drift_cor;
		drift_inc_t[n] = transf_drift_inc;
		threshold_t[n] = transf_threshold;
		ndt_t[n] = transf_ndt;
	}
}

model {
	ndt ~  normal(ndt_priors[1], ndt_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	drift_cor ~ normal(drift_priors[1], drift_priors[2]);
	drift_inc ~ normal(drift_priors[1], drift_priors[2]);

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
