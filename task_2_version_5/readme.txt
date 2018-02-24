Here presented short discriptions about .py and .ipynb files:

1.data_import.py: import the data from the TIMIT set and store in a dictionary

2.main_feature_eng.py: combined all the feature engineering function, import original data and output 15-dimensional featres

3.frame_func.py: frame segmentation and voice activity detection

4.window_func.py: apply window function on the frames

5.mel_func.py: frame extraction, output 22*frames data

6.DCT_func.py: DCT transfer and output features

7.main_crossvalidation.py: combined modeling and identification based on cross validation on 170 speakers

8.density_func.pyx : cython files, describe Probability density function

9.setup: cython scripts to transform .pyx to .c

10.Speaker_model.py: build up models for each speaker

11.Speaker_identification.py: identified for the target test set

12.plot_confusion_matrix.py: plot_confusion_matrix

13.enhancement.py/Enhancement.ipynb: implement and evalution of open-set speaker identification

14.process_bar.py : plot the process of running code directly

15.Plot_gmm_distribution: generate the pictures from different covariance types

16.tuning_gamma.ipynb: tunning the signal noise ratio based on crossvalidation

17.tuning_convergence: tunning the iterations of EM algorithm noise ratio based on crossvalidation

18.tuning_covariance_type: tunning the covariance type of GMM based on crossvalidation

