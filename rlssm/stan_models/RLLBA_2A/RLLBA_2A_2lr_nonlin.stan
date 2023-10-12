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
  int<lower=1> N;                 // number of data items
  int<lower=1> K;               // number of options
  real initial_value;
  array[N] int<lower=1> block_label;          // block label
  array[N] int<lower=1> trial_block;          // trial within block

  vector[N] f_cor;                // feedback correct option
  vector[N] f_inc;                // feedback incorrect option

  array[N] int<lower=1, upper=K> cor_option;      // correct option
  array[N] int<lower=1, upper=K> inc_option;      // incorrect option

  array[N] int<lower=1,upper=2> accuracy;       // 1-> correct, 2->incorrect
  array[N] int<lower=0, upper=1> feedback_type; // feedback_type = 0 -> full feedback, feedback_type = 1 -> partial feedback

  array[N] real<lower=0> rt;              // rt

  vector[2] alpha_pos_priors;           // mean and sd of the alpha_pos prior
  vector[2] alpha_neg_priors;           // mean and sd of the alpha_neg prior
  vector[2] ndt_priors;
  vector[2] k_priors;
  vector[2] sp_trial_var_priors;
  vector[2] slop_priors;
  vector[2] drift_asym_priors;      // mean and sd of the prior for asymtot modulation
  vector[2] drift_scaling_priors;     // mean and sd of the prior for scaling
  vector[2] drift_variability_priors;
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
  real alpha_pos;
  real alpha_neg;
  real ndt;
  real k;
  real sp_trial_var;
  real slop;
  real drift_asym;
  real drift_scaling;    // scaling
  real drift_variability;
}


transformed parameters {
  vector<lower=0> [N] ndt_t;         // trial-by-trial ndt
  vector<lower=0> [N] k_t;            // trial-by-trial
  vector<lower=0> [N] sp_trial_var_t;           // trial-by-trial
  vector<lower=0> [N] drift_cor_t;        // trial-by-trial drift rate for predictions
  vector<lower=0> [N] drift_inc_t;        // trial-by-trial drift rate for predictions
  vector<lower=0> [N] drift_variability_t;

  real PE_cor;      // prediction error correct option
  real PE_inc;      // prediction error incorrect option
  vector[K] Q;

  real Q_mean;
  real Q_min;
  real Q_mean_pres[N];              // mean Q presented options

  real<lower=0, upper=1> transf_alpha_pos;
  real<lower=0, upper=1> transf_alpha_neg;
  real<lower=0> transf_ndt;
  real<lower=0> transf_k;
  real<lower=0> transf_sp_trial_var;
  real<lower=0> transf_slop;
  real<lower=0> transf_drift_asym;
  real<lower=0> transf_drift_scaling;
  real<lower=0> transf_drift_variability;

  transf_alpha_pos = Phi(alpha_pos);
  transf_alpha_neg = Phi(alpha_neg);
  transf_ndt = log(1 + exp(ndt));
  transf_k = log(1 + exp(k));
  transf_sp_trial_var = log(1 + exp(sp_trial_var));
  transf_slop = log(1 + exp(slop));
  transf_drift_asym = log(1 + exp(drift_asym));
  transf_drift_scaling = log(1 + exp(drift_scaling));
  transf_drift_variability = log(1 + exp(drift_variability));

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

    drift_cor_t[n] = (transf_drift_scaling + 0.1*transf_drift_asym*(Q_mean_pres[n] - Q_min)) / (1+exp(transf_slop*(Q_mean_pres[n] - Q[cor_option[n]])));
    drift_inc_t[n] = (transf_drift_scaling + 0.1*transf_drift_asym*(Q_mean_pres[n] - Q_min)) / (1+exp(transf_slop*(Q_mean_pres[n] - Q[inc_option[n]])));

    ndt_t[n] = transf_ndt;
    k_t[n] = transf_k;
    sp_trial_var_t[n] = transf_sp_trial_var;
    drift_variability_t[n] = transf_drift_variability;

    if (PE_cor >= 0) {
      Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_pos*PE_cor;
    } else {
      Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_neg*PE_cor;
    }
    if (PE_inc >= 0) {
      Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_pos*PE_inc;
    } else {
      Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_neg*PE_inc;
    }

    if (feedback_type[n] == 1){
      if(accuracy[n] == 1){
        if (PE_cor >= 0) {
          Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_pos*PE_cor;
        } else {
          Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_neg*PE_cor;
        }
      }
      else{
        if (PE_inc >= 0) {
          Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_pos*PE_inc;
        } else {
          Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_neg*PE_inc;
        }
      }
    }
    else{
      if (PE_cor >= 0) {
        Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_pos*PE_cor;
      } else {
        Q[cor_option[n]] = Q[cor_option[n]] + transf_alpha_neg*PE_cor;
      }
      if (PE_inc >= 0) {
        Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_pos*PE_inc;
      } else {
        Q[inc_option[n]] = Q[inc_option[n]] + transf_alpha_neg*PE_inc;
      }
    }
  }
}

model {
  alpha_pos ~ normal(alpha_pos_priors[1], alpha_pos_priors[2]);
  alpha_neg ~ normal(alpha_neg_priors[1], alpha_neg_priors[2]);
  ndt ~ normal(ndt_priors[1], ndt_priors[2]);
  k ~ normal(k_priors[1], k_priors[2]);
  sp_trial_var ~ normal(sp_trial_var_priors[1], sp_trial_var_priors[2]);
  slop ~ normal(slop_priors[1], slop_priors[2]);
  drift_asym ~ normal(drift_asym_priors[1], drift_asym_priors[2]);
  drift_scaling ~ normal(drift_scaling_priors[1], drift_scaling_priors[2]);
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
