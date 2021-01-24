
data{
	int<lower=1> N;
	int<lower=1> K;
	matrix [N, K] feedback;
	int<lower=1, upper=K> choice[N];
  int<lower=1> block_label[N];
	real initial_value;
}

transformed data {
	vector[K] Q0;
	Q0 = rep_vector(initial_value, K);
}

parameters {
	real alpha;
	real sensitivity;
}

transformed parameters {    
  vector[K] p[N];
	vector[K] Q;
  
  real Q_mean;
	
	real transf_alpha;
	real transf_sensitivity;
	real theta;

	transf_alpha = Phi(alpha);
	transf_sensitivity = log(1 + exp(sensitivity));
	
	for (n in 1:N) {
  
    if (block_label[n] == 1 && n == 1){
      Q = Q0;
    }
    else if (block_label[n] != block_label[n-1]){
      
      Q_mean = mean(Q);
      Q = rep_vector(Q_mean, K);
    }

		p[n] = softmax(Q * transf_sensitivity); 
    for (j in 1:K)
			Q[j] = Q[j] + transf_alpha*(feedback[n, j] - Q[j]);
	}
}

model{
	alpha ~ normal(0, 1);
	sensitivity ~ normal(0, 5);
    
    for (n in 1:N) {
        choice[n] ~ categorical(p[n]);
    }
}