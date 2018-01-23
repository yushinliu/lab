import numpy as np
from Speaker_model import naive_G_U

#math
import math

K_value=49

def Speaker_identification(b_test,new_mu,new_cov,new_weight,T_value):
	 #caculate the concatenated probability
	 test_pdf=naive_G_U(b_test,new_mu,new_cov,new_weight,T_value)
	 #print(test_pdf)
	 scores=np.sum(np.log(test_pdf))
	 return scores
