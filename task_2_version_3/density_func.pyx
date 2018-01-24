import numpy as np
from cpython cimport array
import array
from scipy.stats import norm,multivariate_normal

#math
#import math

cdef int K_value=49

# probability density function
def density_func(b_train,ubm_means,ubm_var,T_value,K_value):
    cdef array.array prob_set=np.zeros((K_value,T_value))
    cdef int k,t
    for k in range(K_value):
        for t in range(T_value):
            prob_set[k,t]=multivariate_normal.pdf(b_train[:,t],ubm_means[k],ubm_var[k,:,:])
    return prob_set