import numpy as np 

def window_func(train_frame_set):
    window = np.hanning(len(train_frame_set[0]))  #different in matlab
    #window = np.hamming(len(train_frame_set[0]))
    train_frame_set=np.float64(train_frame_set)
    frame_windowed_set=np.multiply(train_frame_set,window) # apply the window to the frames
    #using np.multiply , multipy by elements
    return frame_windowed_set