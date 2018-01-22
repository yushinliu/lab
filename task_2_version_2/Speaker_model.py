import numpy as np

#math
import math

K_value=49

def Speaker_model(b_train,ubm_weights,ubm_means,ubm_var,ubm_value_set,T_value,gamma_UBM):
	posteri_prob=post_prob_model(b_train,ubm_weights,ubm_means,ubm_var,ubm_value_set,T_value)
	new_mu=mu_model(posteri_prob,b_train)
	new_cov=cov_model(posteri_prob,b_train,new_mu,T_value)
	new_weight=weight_model(posteri_prob,T_value)
	new_mu_adapted,new_cov_adapted,new_weight_adapted=adapted_mode(posteri_prob,new_mu,ubm_means,new_cov,ubm_var,new_weight,ubm_weights,gamma_UBM)
	return new_mu_adapted,new_cov_adapted,new_weight_adapted





# probability density function
def density_func(b_train,ubm_means,ubm_var,T_value_1,K_value_1):
    D=b_train.shape[0]
    prob=(1/((2*math.pi)**(D/2)*(np.linalg.det(ubm_var[K_value_1,:,:]))**(1/2)))*\
    np.exp((-0.5)*np.dot(np.dot((b_train[:,T_value_1]-ubm_means[K_value_1]).T,np.linalg.inv(ubm_var[K_value_1,:,:])),(b_train[:,T_value_1]-ubm_means[K_value_1])))
    return prob

#calculate the naive GMM-UBM
def naive_G_U(b_train,ubm_means,ubm_var,ubm_weights,T_value):
    ubm_value_set=[]
    for t in range(T_value): 
        ubm_pdf=0
        for i in range(K_value):                                                      
            ubm_pdf+=ubm_weights[i]*density_func(b_train,ubm_means,ubm_var,t,i)
        #print(ubm_pdf)
        ubm_value_set.append(ubm_pdf)
        #prob_ubm=np.array(prob_ubm)
       # ubm_value = np.dot(ubm_weights,prob_ubm)
        #ubm_value = np.dot(ubm_weights,prob_ubm).flatten()
    ubm_value_set=np.array(ubm_value_set)
    return ubm_value_set

#calculate the posteri_prob
def post_prob_model(b_train,ubm_weights,ubm_means,ubm_var,ubm_value_set,T_value):
    posteri_prob=np.zeros((K_value,T_value)) #include 49 models
    for k in range(K_value):
        for t in range(T_value):
            posteri_prob[k,t]=ubm_weights[k]*density_func(b_train,ubm_means,ubm_var,t,k)/ubm_value_set[t]
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
            b_temp=np.dot(b_new[:,t].reshape(-1,1),b_new[:,t].reshape(1,-1))
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

