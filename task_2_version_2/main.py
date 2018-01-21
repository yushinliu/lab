from data_import import data_import

audio_path = "D:\\LAB\\workspace\\lab\\patRecDat\\forStudents\\timit\\test"
#audio_path = "/Users/Mata/Documents/2017/学习/ws2017:18/PUL/forStudents/timit/test"

dataset=data_import(audio_path)  #samples is a dictionary of 172 persons

feature_set={}

for name in dataset.keys():
	print("make the feature of "+ name)
	samples=dataset.get(name,'no such file name') # samples of one person

    frames=frame_func(samples)  #frames is a list with length of 320*frames

	window_frames=window_func(frames) #using hanning window

	freq_frames=mel_func(window_frames)# 

	features=DCT_func(freq_frames)# features: 15*frames

	feature_set.setdefault(name)

	feature_set[name]=features





