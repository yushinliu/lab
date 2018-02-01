from data_import import data_import
from frame_func import frame_func
from window_func import window_func
from mel_func import mel_func
from DCT_func import DCT_func
from process_bar import ShowProcess

#default 
import pickle
import numpy as np
#loadmat
from scipy.io import loadmat,savemat

t_feed=10 #feed time
t_frame=20 #frame time
sample_rate=16000
fs=sample_rate/1000 #sample_rate of each ms
L_value=np.int(fs*t_frame)
NFFT=512
nfilt=22

audio_path = "D:\\LAB\\workspace\\lab\\patRecDat\\forStudents\\timit\\test"
#audio_path = "/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/timit/test"

dataset=data_import(audio_path)  #samples is a dictionary of 172 persons

feature_all_set={}
print("feature engineering start")
process_bar=ShowProcess(len(dataset.keys()))
for name in dataset.keys():
	process_bar.show_process()
	#print("make the feature of "+ name)
	single_data=dataset.get(name,'no such file name') # samples of one person
	features_set=[]
	for samples in single_data:
		if name in ['yuxin','qianqian','shanqi']: #custom voice has dimension error 
			samples=samples[:,0]
		else:
			pass

		frames=frame_func(samples)  #frames is a list with length of 320*frames

		window_frames=window_func(frames) #using hanning window

		freq_frames=mel_func(window_frames)# 

		features=DCT_func(freq_frames)# features: 15*frames

		#features=features.ravel() # need to be reshape when import again

		features_set.append(features)

	feature_all_set.setdefault(name)

	feature_all_set[name]=features_set

save_path="D:\\LAB\\lab\\task_2_version_2\\features.txt"
f = open(save_path,'wb')
pickle.dump(feature_all_set,f)
f.close()
#savemat(save_path,feature_all_set)  
process_bar.close()
print("feature extraction compleleted")





