from data_import import data_import
from frame_func import frame_func
from window_func import window_func
from mel_func import mel_func
from DCT_func import DCT_func
from process_bar import ShowProcess
from Speaker_model import Speaker_model

#default

import numpy as np
#loadmat
from scipy.io import loadmat,savemat

#audio_path = "D:\\LAB\\workspace\\lab\\patRecDat\\forStudents\\timit\\test"
audio_path = "/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/timit/test"

#ubm_dataset=loadmat("/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/ubm/UBM_GMMNaive_MFCC_Spectrum0to8000Hz.mat",mat_dtype=True)
ubm_dataset=loadmat("C:\\Users\\hasee\\workspace\\workspace\\lab\\patRecDat\\forStudents\\ubm\\UBM_GMMNaive_MFCC_Spectrum0to8000Hz.mat",mat_dtype=True)
ubm_means=ubm_dataset['means']
ubm_var = ubm_dataset['var']
ubm_weights = ubm_dataset['weights'].ravel()
ubm_var_set=[]
K_value=49
gamma_UBM=0.001

#transfer variance of UBM to cov
for k in range(K_value):
    ubm_var_set.append(np.diag(ubm_var[k]))
ubm_var_set=np.array(ubm_var_set)
ubm_var=ubm_var_set

dataset=data_import(audio_path)  #samples is a dictionary of 172 persons

features=loadmat("/Users/Mata/Documents/lab/features.mat")
score_set=[]
correct_num=0
false_num=0
name_set=[]
#modul_set={}

for name in features.keys():
    feature_all_set.setdefault(name)
    name_set.append(name)



print("crossvalidation start")
for cross_num in range 10:
    name_max=0
    scores_max=0
    for name in features.keys():
        ubm_value_set=[]
        whole_set=features.get(name,'no such file').copy()
        whole_set=whole_set.reshape(15,-1)
        
        test_file=whole_set[cross_num]
        #test_file_set[name].append(test_file)
        #name_set.append(name)
        train_set=whole_set.copy()
        train_set.remove(test_file)
        train_set=np.concatenate(train_set)
        #train_file_set[name].append(train_set)
        
        test_set=np.array(test_file)
        K_value=ubm_var.shape[0] #number of naive models
        T_value=test_set.shape[1]
        eva_set=[]
        
        for name_2 in features.keys():
            
        
        #5.1 naive GMM-UBM
        ubm_value_set=naive_G_U(train_set,ubm_means,ubm_var,ubm_weights,T_value)
        #5.2 Speaker adaption
        new_mu,new_cov,new_weight=Speaker_model(train_set,ubm_weights,ubm_means,ubm_var,ubm_value_set,T_value,gamma_UBM)
        scores=identification(test_file,new_mu,new_cov,new_weight,T_value)
        if scores>scores_max:
           name_max = name
    
        
        
            if name_set[index_2] == test_name:
        correct_num +=1
else:
    false_num +=1



