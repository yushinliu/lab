import pandas as pd

import numpy as np 
np.set_printoptions(suppress=False) 

#default 
t_feed=10 #feed time
t_frame=20 #frame time
sample_rate=16000
fs=sample_rate/1000 #sample_rate of each ms
L_value=np.int(fs*t_frame)
NFFT=512
nfilt=22


def frame_segment(samples):
    #frame_num = K 
    frame_num=np.int((len(samples)-L_value)/(fs*t_feed))+1 #calculate the number of frames
    frame_set=[]
    for k in range(frame_num):
        frame_set.append(samples[k*np.int(fs*t_feed):k*np.int(fs*t_feed)+L_value])#[0,n] contains 0~n-1
    return frame_set,frame_num

	#combined function
def voice_activity_detection(frame_set,frame_num,gamma=10):
    #step1:figure out the noise signal power
    mixed_power_set=[]
    for k in range(frame_num):
        mixed_power_set.append(1/(L_value)*np.sum((np.float64(frame_set[k]))**2))#use np.float64 to avoid overflow encountered in long scalars
    #step2:The noise power
    t_n=100
    frame_drop=np.int((t_n/t_feed))
    no_speech_set=mixed_power_set[:frame_drop-1]
    noise_power_set=(1/frame_drop)*np.sum(no_speech_set)
    #step3: detective
    mixed_power_set=pd.Series(mixed_power_set)
    frame_set=pd.Series(frame_set)
    train_power_set=frame_set.loc[mixed_power_set>gamma*noise_power_set]
    train_frame_set=np.array(list(train_power_set))
    return train_frame_set
def frame_func(samples,gamma):
	frame_set,frame_num=frame_segment(samples)
	train_frame_set=voice_activity_detection(frame_set,frame_num,gamma)
	return train_frame_set

