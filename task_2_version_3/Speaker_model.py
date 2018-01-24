import numpy as np
from scipy.stats import norm,multivariate_normal
from density_func import density_func


#math
import math

K_value=49

def Speaker_model(b_train,ubm_weights,ubm_means,ubm_var,T_value,gamma_UBM):
	prob_set=density_func(b_train,ubm_means,ubm_var,T_value,K_value)
	ubm_value_set=naive_G_U(prob_set,ubm_weights)
	posteri_prob=post_prob_model(prob_set,ubm_weights,ubm_value_set)
	new_mu=mu_model(posteri_prob,b_train)
	new_cov=cov_model(posteri_prob,b_train,new_mu,T_value)
	new_weight=weight_model(posteri_prob,T_value)
	new_mu_adapted,new_cov_adapted,new_weight_adapted=adapted_mode(posteri_prob,new_mu,ubm_means,new_cov,ubm_var,new_weight,ubm_weights,gamma_UBM)
	return new_mu_adapted,new_cov_adapted,new_weight_adapted






#calculate the naive GMM-UBM
def naive_G_U(prob_set,ubm_weights):                                                    
    ubm_value_set=np.dot(ubm_weights,prob_set)
    return ubm_value_set

#calculate the posteri_prob
def post_prob_model(prob_set,ubm_weights,ubm_value_set):
    prob=np.multiply(ubm_weights.reshape(49,1),prob_set)
    posteri_prob=prob/ubm_value_set
    return posteri_prob


def mu_model(posteri_prob,b_train):
    #b_train: features*frames
    #posteri_prob:models*frames
    value_temp=np.dot(posteri_prob,b_train.T)
    new_mu=np.multiply((1/np.sum(posteri_prob,axis=1)),value_temp.T)
    return new_mu.T

def cov_model(posteri_prob,b_train,new_mu,T_value):
    #new_mu: models*features
    #b_train: features*frames
    #posteri_prob:models*frames
    cov_set =[]
    #calculate mu*mu.T
    for k in range(K_value):
        mu_temp=np.dot(new_mu[k,:].reshape(-1,1),new_mu[k,:].reshape(1,-1))
        #print(mu_temp.shape)
        value_temp=1/np.sum(posteri_prob[k,:])
        sum_temp=0
        for t in range(T_value):
            b_temp=np.dot(b_train[:,t].reshape(-1,1),b_train[:,t].reshape(1,-1))
            #print(b_temp.shape)
            sum_temp+=posteri_prob[k,t]*b_temp
        #print(np.diag(value_temp*sum_temp-mu_temp).shape)
        cov_set.append(np.diag(np.diag(value_temp*sum_temp-mu_temp)))
    cov_set=np.array(cov_set)
    return cov_set

def weight_model(posteri_prob,T_value):
    return (1/T_value)*np.sum(posteri_prob,axis=1)

def adapted_mode(posteri_prob,new_mu,ubm_means,new_cov,ubm_var,new_weight,ubm_weights,gamma_UBM):
    
    #caculate alpha
    alpha = np.sum(posteri_prob,axis=1)/(gamma_UBM+np.sum(posteri_prob,axis=1))

    #caculate the adapted mean
    new_mu_adapted=np.multiply(alpha,new_mu.T)+np.multiply((1-alpha),ubm_means.T)

    #calculate adapted variance    
    new_cov_adapted=np.multiply(alpha,new_cov.T)+np.multiply((1-alpha),ubm_var.T)
    #calculate adapted mean
    new_weight_adapted=np.multiply(alpha,new_weight)+np.multiply((1-alpha),ubm_weights.ravel())
    return new_mu_adapted.T,new_cov_adapted.T,new_weight_adapted.ravel()

