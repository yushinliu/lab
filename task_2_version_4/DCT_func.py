num_features=15
import numpy as np
#math
import math

def DCT_func(Y):
    b_set=[]
    for n_value in range(num_features): 
        b_n =0
        for m in range(Y.shape[0]):
            b_n+=np.multiply(np.log10(Y[m]),np.cos(math.pi*(n_value+1)*(m-0.5)/(Y.shape[0])))#ignore the mean a
            #abandon the n=0
        b_set.append(b_n)
    b_set=np.array(b_set)
    return b_set