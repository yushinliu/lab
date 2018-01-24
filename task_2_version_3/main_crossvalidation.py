
from process_bar import ShowProcess
from Speaker_model import Speaker_model
from Speaker_identification import Speaker_identification
from Speaker_model import naive_G_U

#default

import pickle
import numpy as np
import time

#loadmat
from scipy.io import loadmat,savemat

#audio_path = "/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/timit/test"

#save_path="D:\\LAB\\lab\\task_2_version_3\\features.txt"
save_path="/Users/Mata/Documents/lab/task_2_version_3/features.txt"
f = open(save_path,'rb')
features=pickle.load(f)
f.close()

ubm_dataset=loadmat("/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/ubm/UBM_GMMNaive_MFCC_Spectrum0to8000Hz.mat",mat_dtype=True)
#ubm_dataset=loadmat("C:\\Users\\hasee\\workspace\\workspace\\lab\\patRecDat\\forStudents\\ubm\\UBM_GMMNaive_MFCC_Spectrum0to8000Hz.mat",mat_dtype=True)
ubm_means=ubm_dataset['means']
ubm_var = ubm_dataset['var']
ubm_weights = ubm_dataset['weights'].ravel()
ubm_var_set=[]
K_value=49
gamma_UBM=1

#transfer variance of UBM to cov
for k in range(K_value):
    ubm_var_set.append(np.diag(ubm_var[k]))
ubm_var_set=np.array(ubm_var_set)
ubm_var=ubm_var_set

num_people=len(features.keys())

#process_bar=ShowProcess(10)
for cross_num in range(10):
	#process_bar.show_process()
	test_file_set=[]
	new_mu_set=[]
	new_cov_set=[]
	new_weight_set=[]
	correct_num=0
	false_num=0
	print("crossvalidation "+str(cross_num+1)+" building model start")
	process_bar_1=ShowProcess(len(features.keys()))
	for name in features.keys():
		start=time.time()
		process_bar_1.show_process()
		ubm_value_set=[]
		whole_set=features.get(name,'no such file').copy()
		test_file=whole_set[cross_num]
		test_file_set.append(test_file)
        #name_set.append(name)
		train_set=whole_set.copy()
		train_set.remove(test_file)
		train_set=np.concatenate(train_set,axis=1)
		T_value=train_set.shape[1]
        #5.2 Speaker adaption
		new_mu,new_cov,new_weight=Speaker_model(train_set,ubm_weights,ubm_means,ubm_var,T_value,gamma_UBM)
		new_mu_set.append(new_mu)
		new_cov_set.append(new_cov)
		new_weight_set.append(new_weight)
		print(start-time.time())
	process_bar_1.close()
	print("crossvalidation "+str(cross_num+1)+" building model compeleted")
	print("crossvalidation "+str(cross_num+1)+" identification start")
	process_bar_2=ShowProcess(num_people)
	for index_1 in range(num_people):
		process_bar_2.show_process()
		T_value=test_file_set[index_1].shape[1]
		scores_set=[]
		for index_2 in range(num_people):
			scores=Speaker_identification(test_file_set[index_1],new_mu_set[index_2],new_cov_set[index_2],new_weight_set[index_2],T_value)
			scores_set.append(scores)
		if index_1==scores_set.index(max(scores_set)):
			correct_num +=1
		else:
			false_num +=1
	process_bar_2.close()
	print("crossvalidation "+str(cross_num+1)+" identification compeleted")
	detection_rate=correct_num/(false_num+correct_num)
	print("the crossval "+str(cross_num)+" detection_rate is "+str(detection_rate))
	
#process_bar.close()
	


