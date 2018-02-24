
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import random
import time
import sys
import itertools
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from calculation_score import cal_score
from data_basic_operation import choose_data_seperately
from data_basic_operation import choose_data_together


# In[2]:


# This is the basic function which encapsalation the result as well as the calculation performance.
def SVM_base_fuction (X_train,y_train,X_val,y_val,C_parameter,gamma_parameter):
    svc_clf=SVC(kernel="rbf",degree=len(X_train),C = C_parameter,gamma = gamma_parameter)
    svc_clf.fit(X_train,y_train['label'])
    y_pred=svc_clf.predict(X_val)
    y_pred = {"label_pred":y_pred}
    y_pred = pd.DataFrame(data=y_pred)
    Precall,f1_score,BER,FPR= cal_score(y_pred,y_val['label'])
    return Precall,f1_score,BER,FPR,y_pred


# In[2]:

# This is used for the final validation step.
def SVM_validation (X_train,y_train,X_val,y_val,sample_amount,data_ratio,C_parameter,gamma_parameter):
    #sample_amount = 40000
    #data_ratio = 1.2
    label_1_amount = int(sample_amount * (data_ratio/(data_ratio+1)))
    label_2_amount = int(sample_amount-label_1_amount)
    tpr_list_temp = []
    fpr_list_temp = []
    BER_list_temp = []
    f1_score_list_temp = []
    iter_max = 4
    count = 0
    while (count < iter_max):
        sample_feature,sample_label = choose_data_seperately (X_train,y_train,label_1_amount,label_2_amount)
        Precall,f1_score,BER,FPR,y_pred=SVM_base_fuction (sample_feature,sample_label,X_val,y_val,2**C_parameter,2**gamma_parameter)
        tpr_list_temp.append(Precall)
        fpr_list_temp.append(FPR)
        BER_list_temp.append(BER)
        f1_score_list_temp.append(f1_score)
        count = count+1
    
    tpr=(sum(tpr_list_temp)/len(tpr_list_temp))
    fpr=(sum(fpr_list_temp)/len(fpr_list_temp))
    BER=(sum(BER_list_temp)/len(BER_list_temp))
    f1_score=(sum(f1_score_list_temp)/len(f1_score_list_temp))
    return tpr,fpr,BER,f1_score
        


# In[3]:

# This is the basic part of adjustment in the training phase withe cross-calisation. The average results will be returned.
def SVM_cross_validation (training_feature, training_label,C_parameter,gamma_parameter):
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(training_feature,training_label['label'])
    tpr_list_temp = []
    fpr_list_temp = []
    BER_list_temp = []
    f1_score_list_temp = []
    time_list_temp = []
    
    for train_index, test_index in skf.split(training_feature,training_label['label']):
        start1=time.time()
        X_train, X_val = training_feature.loc[train_index], training_feature.loc[test_index]
        y_train, y_val = training_label.loc[train_index], training_label.loc[test_index]

        Precall,f1_score,BER,FPR,y_pred = SVM_base_fuction(X_train,y_train,X_val,y_val,C_parameter,gamma_parameter)

        tpr_list_temp.append(Precall)
        fpr_list_temp.append(FPR)
        BER_list_temp.append(BER)
        f1_score_list_temp.append(f1_score)
        time_list_temp.append(time.time()-start1)

    
    tpr=(sum(tpr_list_temp)/len(tpr_list_temp))
    fpr=(sum(fpr_list_temp)/len(fpr_list_temp))
    BER=(sum(BER_list_temp)/len(BER_list_temp))
    f1_score=(sum(f1_score_list_temp)/len(f1_score_list_temp))
    time_var=(sum(time_list_temp)/len(time_list_temp))
    return tpr,fpr,BER,f1_score,time_var


# In[4]:

# Have an adjustment with the sample amount and the result will be returned as a data frame.
def sample_amount_choice (X_train,y_train,sample_amount_upper_bound):
    start = time.time()
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    sample_amount_list=[]
    sample_amount =sample_amount_upper_bound;
    iter_amount = 3
    
    while (sample_amount>800):
        count=0
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]
        
        while (count<iter_amount):
            print ("current sample amount:%d"%sample_amount)
            
            sample_feature,sample_label = choose_data_together (X_train,y_train,sample_amount)
    
            tpr,fpr,BER,f1_score,time_var = SVM_cross_validation (sample_feature, sample_label,1,'auto')
            
            time_list_temp.append(time_var)
            tpr_list_temp.append(tpr)
            fpr_list_temp.append(fpr)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            count = count+1       
        
        sample_amount_list.append(sample_amount)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        if (sample_amount >10000):
            sample_amount=int(sample_amount/2)
        elif (sample_amount<=2000 ):
            sample_amount=sample_amount-100
            iter_amount=10
        else:
            sample_amount=sample_amount-2000
            iter_amount=10
    
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))
    result = {"sample_amount":sample_amount_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["sample_amount","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[5]:

# Have an adjustment with the data ratio and the result will be returned as a data frame.
def SVC_data_ratio_adjust (X_train,y_train,sample_amount):
    start = time.time()
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    label_1_amount_list = []
    label_2_amount_list = []
    ratio_list = []
   
    iter_amount = 5
    data_ratio = 4
    
    while (data_ratio > 0.2):
        count=0
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]
        label_1_amount = int(sample_amount * (data_ratio/(data_ratio+1)))
        label_2_amount = int(sample_amount-label_1_amount)
        while (count<iter_amount):
            start1=time.time()

            sample_feature,sample_label = choose_data_seperately (X_train,y_train,label_1_amount,label_2_amount)            
            
            tpr,fpr,BER,f1_score,time_var= SVM_cross_validation (sample_feature, sample_label,1,'auto')
            
            time_list_temp.append(time_var)
            tpr_list_temp.append(tpr)
            fpr_list_temp.append(fpr)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            count = count+1   
       
        label_1_amount_list.append(label_1_amount)
        label_2_amount_list.append(label_2_amount)
        ratio_list.append(label_1_amount/label_2_amount)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        
        if (data_ratio>2):
            data_ratio = data_ratio / 2
        elif (data_ratio<0.8):
            data_ratio = data_ratio-0.25
        else:
            data_ratio = data_ratio -0.1
               
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))       
    result = {"label_1_amount":label_1_amount_list,"label_2_amount":label_2_amount_list,"label 1: label 2 ratio":ratio_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["label_1_amount","label_2_amount","label 1: label 2 ratio","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[7]:

# Have an adjustment with the C and gamma exp and the results will be returned as a data frame.
def parameter_adjust (X_train,y_train,sample_amount,data_ratio):
    start = time.time()
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    gamma_exp_list=[]
    C_exp_list=[]
    label_1_amount = int(sample_amount * (data_ratio/(data_ratio+1)))
    label_2_amount = int(sample_amount-label_1_amount)
     
    for gamma_exp in [-15,-13,-11,-9,-7,-5,-3,-1,1,3]:
        for C_exp in [-5,-3,-1,1,3,5,7,9,11,13,15]:
            start1=time.time();
            
            sample_feature,sample_label = choose_data_seperately (X_train,y_train,label_1_amount,label_2_amount)                        
            
            tpr,fpr,BER,f1_score,time_var = SVM_cross_validation (sample_feature, sample_label,2**C_exp,2**gamma_exp)
           
            time_list.append(time_var)
            tpr_list.append(tpr)
            f1_score_list.append(f1_score)
            BER_list.append(BER)
            fpr_list.append(fpr)
            gamma_exp_list.append(gamma_exp)
            C_exp_list.append(C_exp)
            #print("fit time:%5.1fminute"%(temp))
    
    print("the total executing time:%5.1fminute"%((time.time()-start)/60)) 
    result = {"gamma_exp":gamma_exp_list,"C_exp":C_exp_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["gamma_exp","C_exp","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result

