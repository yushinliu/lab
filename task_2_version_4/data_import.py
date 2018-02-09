#functin data_import:
import os
from os.path import isdir, join
import csv
import soundfile as sf

def data_import(audio_train_path,unknown):
	if unknown ==0 :     # extract registered features
		Name_set={}
		test_set=[]
		count =0
		for area in os.listdir(audio_train_path):#os.listdit: to show the files in this path
			#for name in os.listdir(audio_train_path+'\\'+area+"\\"):
			if (area=='.DS_Store'):
				continue
			else:
				for name in os.listdir(audio_train_path+'/'+area+"/"):
					if (name =='.DS_Store'):
						continue
					else:
						Name_set.setdefault(name)  #add new key to the dict
						sample_set=[]
						#print (name)
						#count +=1
						#print (count)
						#for files in os.listdir(audio_train_path+'\\'+area+'\\'+name+'\\'):
						for files in os.listdir(audio_train_path+'/'+area+'/'+name+'/'):
							#_,samples=wavfile.read(audio_train_path+'\\'+area+'\\'+name+'\\'+files)#read the wavfile , return sample_rate ,and samples
							samples,_=sf.read(audio_train_path+'/'+area+'/'+name+'/'+files)
							sample_set.append(samples)
							Name_set[name]=sample_set
	else:               # extract unregistered( unknown ) features
		Name_set={}
		test_set=[]
		count =0
		for area in range(3):#os.listdit: to show the files in this path
			for name in os.listdir(audio_train_path+'\\dr'+str(area+1)+"\\"):
			#for name in os.listdir(audio_train_path+'/dr'+str(area)+"/"):
				if (name =='.DS_Store'):
					continue
				else:
					Name_set.setdefault(name)  #add new key to the dict
					sample_set=[]
					#print (name)
					#count +=1
					#print (count)
					for files in os.listdir(audio_train_path+'\\dr'+str(area+1)+'\\'+name+'\\'):
					#for files in os.listdir(audio_train_path+'/dr'+str(area)+'/'+name+'/'):
						#_,samples=wavfile.read(audio_train_path+'\\'+area+'\\'+name+'\\'+files)#read the wavfile , return sample_rate ,and samples
						samples,_=sf.read(audio_train_path+'\\dr'+str(area+1)+'\\'+name+'\\'+files)
						sample_set.append(samples)
						Name_set[name]=sample_set
	return Name_set