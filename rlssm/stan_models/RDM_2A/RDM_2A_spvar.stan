functions {
     real race_pdf(real t, real b, real v, real A){
          real pdf;
          real aa;
          real bb;
          aa = (b - A - t*v)/sqrt(t);
          bb = (b - t*v)/sqrt(t);
          pdf = (-v*Phi(aa) + exp(std_normal_lpdf(aa))/sqrt(t) + v*Phi(bb) - exp(std_normal_lpdf(bb))/sqrt(t))/A;
          return pdf;
     }

     real race_cdf(real t, real b, real v, real A){
          real cdf;
          real a1;
          real a2;
          real b1;
          real b2;
          
          real t1;
          real t2;
          real t3;
          real t4;

          a1 = sqrt(2) * (v*t - b)/sqrt(2*t);
          b1 = sqrt(2) * (-v*t - b)/sqrt(2*t);

          a2 = sqrt(2) * (v*t - (b-A))/sqrt(2*t);
          b2 = sqrt(2) * (-v*t - (b-A))/sqrt(2*t);

          t1 = (Phi(a2) - Phi(a1))/(2*v*A);
          t2 = sqrt(t)*(a2*Phi(a2) - a1*Phi(a1))/A;
          t3 = (exp(2*v*(b-A))*Phi(b2) - exp(2*v*b)*Phi(b1))/(2*v*A);
          t4 = sqrt(t)*(exp(std_normal_lpdf(a2)) - exp(std_normal_lpdf(a1)))/A;

          cdf = t1 + t2 - t3 + t4;
          return cdf;
     }

     real race_lpdf(matrix RT, vector  ndt, vector b, vector drift_cor, vector drift_inc, vector sp_trial_var){

          real t;
          vector[rows(RT)] prob;
          real cdf;
          real pdf;
          real out;

          for (i in 1:rows(RT)){
               t = RT[i,1] - ndt[i];
               if(t > 0){
                  if(RT[i,2] == 1){
                    pdf = race_pdf(t, b[i], drift_cor[i], sp_trial_var[i]);
                    cdf = 1 - race_cdf(t, b[i], drift_inc[i], sp_trial_var[i]);
                  }
                  else{
                    pdf = race_pdf(t, b[i], drift_inc[i], sp_trial_var[i]);
                    cdf = 1 - race_cdf(t, b[i], drift_cor[i], sp_trial_var[i]);
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
    vector[2] sp_trial_var_priors;
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
	real drift_cor;
	real drift_inc;
    real threshold;
    real sp_trial_var;
}

transformed parameters {
    vector<lower=0> [N] ndt_t;						// trial-by-trial ndt
	vector<lower=0> [N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0> [N] threshold_t;				// trial-by-trial threshold
    vector<lower=0> [N] sp_trial_var_t;

    real<lower=0> transf_ndt;
	real<lower=0> transf_drift_cor;
	real<lower=0> transf_drift_inc;
	real<lower=0> transf_threshold;
    real<lower=0> transf_sp_trial_var;

    transf_ndt = log(1 + exp(ndt));
	transf_drift_cor = log(1 + exp(drift_cor));
	transf_drift_inc = log(1 + exp(drift_inc));
	transf_threshold = log(1 + exp(threshold));
    transf_sp_trial_var = log(1 + exp(sp_trial_var));

	for (n in 1:N) {
        ndt_t[n] = transf_ndt;
		drift_cor_t[n] = transf_drift_cor;
		drift_inc_t[n] = transf_drift_inc;
		threshold_t[n] = transf_threshold + transf_sp_trial_var;
		sp_trial_var_t[n] = transf_sp_trial_var;
	}
}

model {
	ndt ~  normal(ndt_priors[1], ndt_priors[2]);
	threshold ~ normal(threshold_priors[1], threshold_priors[2]);
	drift_cor ~ normal(drift_priors[1], drift_priors[2]);
	drift_inc ~ normal(drift_priors[1], drift_priors[2]);
    sp_trial_var ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);

	RT ~ race(ndt_t, threshold_t, drift_cor_t, drift_inc_t, sp_trial_var_t);
}

generated quantities {
	vector[N] log_lik;
	{
	for (n in 1:N){
		log_lik[n] = race_lpdf(block(RT, n, 1, 1, 2)| segment(ndt_t, n, 1), segment(threshold_t, n, 1), segment(drift_cor_t, n, 1), segment(drift_inc_t, n, 1), segment(sp_trial_var_t, n, 1));
	}
	}
}
