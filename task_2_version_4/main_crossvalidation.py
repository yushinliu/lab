
from process_bar import ShowProcess
from sklearn.mixture import GaussianMixture
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
#default

import pickle
import numpy as np
import time

#loadmat
from scipy.io import loadmat,savemat

'''
Import features
'''

save_path="D:\\LAB\\lab\\task_2_version_3\\features.txt"
#save_path="/Users/Mata/Documents/lab/task_2_version_3/features.txt"
f = open(save_path,'rb')
features=pickle.load(f)
f.close()

'''
Import UBM model
'''

#ubm_dataset=loadmat("/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/ubm/UBM_GMMNaive_MFCC_Spectrum0to8000Hz.mat",mat_dtype=True)
ubm_dataset=loadmat("C:\\Users\\hasee\\workspace\\workspace\\lab\\patRecDat\\forStudents\\ubm\\UBM_GMMNaive_MFCC_Spectrum0to8000Hz.mat",mat_dtype=True)
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

'''
Start Crossvalidation
'''
#process_bar=ShowProcess(10)
detection_rate_set=[]
start=time.time()
error_set={}  
num_samples=len(features.keys())
name_set=list(features.keys())
confusion_matrix=np.zeros((num_samples,num_samples))
correct_sum=0
false_sum=0
for cross_num in range(10):
	#process_bar.show_process()
	train_file_set=[]
	test_file_set=[]
	correct_num=0
	false_num=0
	'''
	Split the test and train set
	'''
	for name in features.keys():
		whole_set=features.get(name,'no such file').copy()
		test_file=whole_set[cross_num]
		train_num=list(range(10))
		train_num.remove(cross_num)
		test_file_set.append(test_file)
        #name_set.append(name)
		train_set=[]
		for num in train_num:
			train_set.append(whole_set[num])
		train_set=np.concatenate(train_set,axis=1)
		train_file_set.append(train_set)
	'''
	Start modeling and identification
	'''
	print("crossvalidation "+str(cross_num+1)+" start")
	process_bar_2=ShowProcess(num_samples)
	scores_set=np.zeros((num_samples,num_samples))
	for index_2 in range(num_samples):
		process_bar_2.show_process()
		b_train=train_file_set[index_2]
		gmm=GaussianMixture(n_components=K_value,covariance_type='full',max_iter=1,weights_init=ubm_weights,\
							means_init=ubm_means,precisions_init=np.linalg.inv(ubm_var))
		gmm.fit(b_train.T)
		for index_1 in range(num_samples):
			#print("now test set "+str(index_1)+" is testing "+str(index_2))
			b_test=np.array(test_file_set[index_1])
			scores_set[index_1,index_2]=gmm.score(b_test.T)

	'''
	Calculate the detection rate
	'''
	for index in range(num_samples):
		test_index=np.int(np.argwhere(scores_set[index,:]==max(scores_set[index,:])))
		if index == test_index:
			correct_num +=1
			correct_sum +=1
			confusion_matrix[index,index] +=1
		else:
			false_num +=1
			false_sum +=1
			confusion_matrix[index,test_index] +=1
			print("error ! True: "+str(name_set[index])+" False: "+str(name_set[test_index]))
		#print("time cost %5.1f second"%((time.time()-start)/60))

	process_bar_2.close()

	detection_rate=correct_num/(correct_num+false_num)
	print("crossvalidation "+str(cross_num+1)+" compeleted")
	print("cost time %5.1f minute"%((time.time()-start)/60))
	detection_rate=correct_num/(false_num+correct_num)
	detection_rate_set.append(detection_rate)
	print("the crossval "+str(cross_num)+" detection_rate is "+str(detection_rate))

print("the total detection rate is ",correct_sum/(correct_sum+false_sum))

save_path="D:\\LAB\\lab\\task_2_version_4\\confusion_matrix.txt"
#save_path="/Users/Mata/Documents/lab/task_2_version_3/features.txt"
f = open(save_path,'wb')
features=pickle.dump(confusion_matrix,f)
f.close()



	


