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
                    cdf = 1 - race_cdf(t| b[i], drift_inc[i], sp_trial_var[i]);
                  }
                  else{
                    pdf = race_pdf(t, b[i], drift_inc[i], sp_trial_var[i]);
                    cdf = 1 - race_cdf(t| b[i], drift_cor[i], sp_trial_var[i]);
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
	array[N] int<lower=1, upper=L> participant;		// level (participant)

	array[N] int<lower=1,upper=2> accuracy;			// 1-> correct, 2->incorrect
	array[N] real<lower=0> rt;						// rt

	vector[4] drift_priors;							// mean and sd of the group mean and of the group sd hyper-priors
	vector[4] threshold_priors;						// mean and sd of the group mean and of the group sd hyper-priors
	vector[4] ndt_priors;							// mean and sd of the group mean and of the group sd hyper-priors
    vector[4] sp_trial_var_priors;
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
    real mu_sp_trial_var;

	real<lower=0> sd_ndt;
	real<lower=0> sd_threshold;
	real<lower=0> sd_drift_cor;
	real<lower=0> sd_drift_inc;
    real<lower=0> sd_sp_trial_var;

	array[L] real z_ndt;
	array[L] real z_threshold;
	array[L] real z_drift_cor;
	array[L] real z_drift_inc;
    array[L] real z_sp_trial_var;
}

transformed parameters {
    vector<lower=0>[N] ndt_t;					// trial-by-trial ndt
	vector<lower=0>[N] drift_cor_t;				// trial-by-trial drift rate for predictions
	vector<lower=0>[N] drift_inc_t;				// trial-by-trial drift rate for predictions
	vector<lower=0>[N] threshold_t;				// trial-by-trial threshold
	vector<lower=0>[N] sp_trial_var_t;

    array[L] real<lower=0> ndt_sbj;
	array[L] real<lower=0> drift_cor_sbj;
	array[L] real<lower=0> drift_inc_sbj;
	array[L] real<lower=0> threshold_sbj;
    array[L] real<lower=0> sp_trial_var_sbj;

    real<lower=0> transf_mu_ndt;
	real<lower=0> transf_mu_drift_cor;
	real<lower=0> transf_mu_drift_inc;
	real<lower=0> transf_mu_threshold;
	real<lower=0> transf_mu_sp_trial_var;

    transf_mu_ndt = log(1 + exp(mu_ndt));
	transf_mu_drift_cor = log(1 + exp(mu_drift_cor));
	transf_mu_drift_inc = log(1 + exp(mu_drift_inc));
	transf_mu_threshold = log(1 + exp(mu_threshold));
    transf_mu_sp_trial_var = log(1 + exp(mu_sp_trial_var));

	for (l in 1:L) {
        ndt_sbj[l] = log(1 + exp(mu_ndt + z_ndt[l]*sd_ndt));
		drift_cor_sbj[l] = log(1 + exp(mu_drift_cor + z_drift_cor[l]*sd_drift_cor));
		drift_inc_sbj[l] = log(1 + exp(mu_drift_inc + z_drift_inc[l]*sd_drift_inc));
		threshold_sbj[l] = log(1 + exp(mu_threshold + z_threshold[l]*sd_threshold));
        sp_trial_var_sbj[l] = log(1 + exp(mu_sp_trial_var + z_sp_trial_var[l]*sd_sp_trial_var));
	}

	for (n in 1:N) {
        ndt_t[n] = ndt_sbj[participant[n]];
		drift_cor_t[n] = drift_cor_sbj[participant[n]];
		drift_inc_t[n] = drift_inc_sbj[participant[n]];
		threshold_t[n] = threshold_sbj[participant[n]] + sp_trial_var_sbj[participant[n]];
        sp_trial_var_t[n] = sp_trial_var_sbj[participant[n]];
	}
}

model {
    mu_ndt ~ normal(ndt_priors[1], ndt_priors[2]);
	mu_drift_cor ~ normal(drift_priors[1], drift_priors[2]);
	mu_drift_inc ~ normal(drift_priors[1], drift_priors[2]);
	mu_threshold ~ normal(threshold_priors[1], threshold_priors[2]);
    mu_sp_trial_var ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);

    sd_ndt ~ normal(ndt_priors[3], ndt_priors[4]);
	sd_drift_cor ~ normal(drift_priors[3], drift_priors[4]);
	sd_drift_inc ~ normal(drift_priors[3], drift_priors[4]);
	sd_threshold ~ normal(threshold_priors[3], threshold_priors[4]);
    sd_sp_trial_var ~ normal(sp_trial_var_priors[3], sp_trial_var_priors[4]);
	
    z_ndt ~ normal(0, 1);
	z_drift_cor ~ normal(0, 1);
	z_drift_inc ~ normal(0, 1);
	z_threshold ~ normal(0, 1);
    z_sp_trial_var ~ normal(0, 1);

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
