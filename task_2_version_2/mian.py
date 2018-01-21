import package.py
import data_import.py

audio_path = "D:\\LAB\\workspace\\lab\\patRecDat\\forStudents\\timit\\test"

dataset=data_import(audio_path)  #samples is a dictionary of 172 persons

for name in dataset.keys():
	print(name)
    

	#frames=frame_func(samples)  #frames is a list with length of 

	#window_frames=window_func(frames)

