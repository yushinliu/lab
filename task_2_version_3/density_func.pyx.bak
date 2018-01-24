import numpy as np
from scipy.stats import norm,multivariate_normal

#math
import math

K_value=49

# probability density function
def density_func(b_train,ubm_means,ubm_var,T_value,K_value):
    prob_set=np.zeros((K_value,T_value))
    for k in range(K_value):
        for t in range(T_value):
            prob_set[k,t]=multivariate_normal.pdf(b_train[:,t],ubm_means[k],ubm_var[k,:,:])
    return prob_set