from frame_func import frame_func
from window_func import window_func
from mel_func import mel_func
from DCT_func import DCT_func
from process_bar import ShowProcess
import soundfile as sf

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

audio_path = "D:\\LAB\\workspace\\lab\\patRecDat\\forStudents\\timit\\train\\dr1\\fcjf0\\sa1.wav"

samples,_=sf.read(audio_path)

frames=frame_func(samples)  #frames is a list with length of 320*frames

window_frames=window_func(frames) #using hanning window

freq_frames=mel_func(window_frames)# 

features=DCT_func(freq_frames)# features: 15*frames

#features=features.ravel() # need to be reshape when import again
print(features.shape)
save_path="D:\\LAB\\lab\\task_2_version_4\\unknown_feature.txt"
f = open(save_path,'wb')
pickle.dump(features,f)
f.close()
#savemat(save_path,feature_all_set)  
print("feature extraction compleleted")





