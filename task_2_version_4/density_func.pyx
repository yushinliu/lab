import numpy as np

from libc.math cimport exp,pi
from cpython cimport array
import array
#from scipy.stats import norm,multivariate_normal

#math
#import math

cdef int K_value=49

# probability density function
def density_func(b_train,ubm_means,ubm_var,T_value,K_value):
    prob_set=np.zeros((K_value,T_value))
    cdef int k,t
    for k in range(K_value):
        for t in range(T_value):
            prob_set[k,t]=(1/((2*pi)**(15/2)*(np.linalg.det(ubm_var[k,:,:]))**(1/2)))*\
    exp((-0.5)*np.dot(np.dot((b_train[:,t]-ubm_means[k]),np.linalg.inv(ubm_var[k,:,:])),(b_train[:,k]-ubm_means[k])))
    return prob_set

  #calculate the naive GMM-UBM
def naive_G_U(prob_set,ubm_weights):                                                    
    ubm_value_set=np.dot(ubm_weights,prob_set)
    return ubm_value_set