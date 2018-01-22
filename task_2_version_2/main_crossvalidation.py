from data_import import data_import
from frame_func import frame_func
from window_func import window_func
from mel_func import mel_func
from DCT_func import DCT_func
from process_bar import ShowProcess

#default

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

#audio_path = "D:\\LAB\\workspace\\lab\\patRecDat\\forStudents\\timit\\test"
audio_path = "/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/timit/test"

dataset=data_import(audio_path)  #samples is a dictionary of 172 persons

feature_all_set={}
score_set=[]
correct_num=0
false_num=0
#whole_set=[]

test_file_set=[]
name_set=[]
train_file_set=[]
features=[]

for name in dataset.keys():
    feature_all_set.setdefault(name)
    test_file_set.setdefault(name)
    train_file_set.setdefault(name)
    name_set.append(name)

print("crossvalidation start")
for cross_num in range 10:
    
    for name in dataset.keys():
        
        whole_set=dataset.get(name,'no such file').copy()
        
        test_file=whole_set[cross_num]
        test_file_set[name].append(test_file)
        #name_set.append(name)
        train_set=whole_set.copy()
        train_set.remove(test_file)
        train_set=np.concatenate(train_set)
        train_file_set[name].append(train_set)

        frames=frame_func(samples)  #frames is a list with length of 320*frames
                
        window_frames=window_func(frames) #using hanning window
                
        freq_frames=mel_func(window_frames)#
                
        features=DCT_func(freq_frames)# features: 15*frames
                
        features=features.ravel() # need to be reshape when import again
        
        #features_set.append(features)
                
        #feature_all_set.setdefault(name)

        feature_all_set[name].append(features_set)

    for name_2,test_file in enumerate(test_file_set):
        test_set=np.array(test_set)
        K_value=ubm_var.shape[0]






