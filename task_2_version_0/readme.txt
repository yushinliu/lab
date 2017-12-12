--------------------------6/12/2017-----------------------------------------
operator: Yuxin Liu
+ build the file

--------------------------7/12/2017-----------------------------------------
operator: Wei Qianqian
+ function"load_data(audio_train_path)" import all the data in the test file and stored them into the dictionary
+ visualize the audio signals
+ function"frame_segment(samples)" segment the audio material into frames
+ add voice detector to separate voiecd from unvoiced frames (initial threshold gamma=2.0)
+ add Von_Hann_Fenster 
+ add Mel Filter Bank (M=22,L=320)


--------------------------8/12/2017-----------------------------------------
operator: Yuxin Liu
+ added the plot of Mel-filter-bank and window

--------------------------9/12/2017-----------------------------------------
operator: Wei Qianqian
+ added Mel Scale Function
+ add function:DCT(Y)

--------------------------10/12/2017-----------------------------------------
operator: Yuxin Liu
+added Covariance function in GMM
+modified the pdf equation from norm to multivariate_normal

--------------------------11/12/2017-----------------------------------------
operator: Wei Qianqian
+added function to calculate posteriori probability,mean and covariance matrix,corresponding weights of the adapted k_th mode of the speaker
+ built the function for voice activity detector

