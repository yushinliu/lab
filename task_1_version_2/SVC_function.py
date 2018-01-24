
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVC
import random
import time
import sys
import itertools
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from calculation_score import cal_score


# In[4]:


# This is the basic function which encapsalation the result as well as the calculation performance.
def SVM_base_fuction (X_train,y_train,X_val,y_val):
    svc_clf=SVC(kernel="rbf",degree=len(X_train))
    svc_clf.fit(X_train,y_train['label'])
    y_pred=svc_clf.predict(X_val)
    y_pred = {"label_pred":y_pred}
    y_pred = pd.DataFrame(data=y_pred)
    Precall,f1_score,BER,FPR= cal_score(y_pred,y_val['label'])
    return Precall,f1_score,BER,FPR,y_pred


# In[3]:


#This function is to reduce the data input for saving time.
def SVM_sample_amount_choice (X_train,y_train,X_val,y_val,sample_amount_upper_bound):
    start = time.time()
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    sample_amount_list=[]
    sample_amount =sample_amount_upper_bound;
    train = pd.concat([X_train,y_train['label']],axis=1,join='outer')
    train = pd.DataFrame(data=train)
    iter_amount = 2
    
    while (sample_amount>800):
        count=0
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]
        while (count<iter_amount):
            #print ("current sample amount:%d"%sample_amount)
            start1=time.time()
            sample =pd.DataFrame.sample(train,sample_amount)
            sample_label = pd.DataFrame(sample['label'])
            sample_feature=sample.drop(["label"],axis=1)

            Precall,f1_score,BER,FPR,y_pred = SVM_base_fuction(sample_feature,sample_label,X_val,y_val) 
            
            temp = (time.time()-start1)/60
            time_list_temp.append(temp)
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
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
        


# In[10]:


def SVC_data_ratio_adjust (X_train,y_train,X_val,y_val,sample_amount):
    start = time.time()
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    label_1_amount_list = []
    label_2_amount_list = []
    ratio_list = []
    train_data = pd.concat([X_train,y_train['label']],axis=1,join='outer')
    train_data = pd.DataFrame(data=train_data)
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
            Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
            Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],label_2_amount)
            res = [Class1_sample, Class2_sample]
            train_com = pd.concat(res)
            #print(train_com)
            sample_label = pd.DataFrame(train_com['label'])
            sample_feature=train_com.drop(["label"],axis=1)
            start1=time.time()

            Precall,f1_score,BER,FPR,y_pred = SVM_base_fuction(sample_feature,sample_label,X_val,y_val) 
            
            temp = (time.time()-start1)/60
            time_list_temp.append(temp)
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
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
    


# In[11]:


def parameter_adjust (X_train,y_train,X_val,y_val,sample_amount,data_ratio):
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
            train_data = pd.concat([X_train,y_train['label']],axis=1,join='outer')
            train_data = pd.DataFrame(train_data)
            Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
            Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],label_2_amount)
            res = [Class1_sample, Class2_sample]
            train_com = pd.concat(res)
            sample_label = pd.DataFrame(train_com['label'])
            sample_feature=train_com.drop(["label"],axis=1)
           
            svc_clf=SVC(kernel="rbf",degree=len(sample_feature),gamma=2**gamma_exp,C=2**C_exp)
            svc_clf.fit(sample_feature,sample_label['label'])
            y_pred=svc_clf.predict(X_val)
            y_pred = {"label_pred":y_pred}
            y_pred = pd.DataFrame(data=y_pred)
            #print ("curren gamma_exp:"+str(gamma_exp))
            #print ("current C_exp:"+str(C_exp))
            Precall,f1_score,BER,FPR= cal_score(y_pred,y_val['label'])
            temp=(time.time()-start1)/60
            time_list.append(temp)
            tpr_list.append(Precall)
            f1_score_list.append(f1_score)
            BER_list.append(BER)
            fpr_list.append(FPR)
            gamma_exp_list.append(gamma_exp)
            C_exp_list.append(C_exp)
            #print("fit time:%5.1fminute"%(temp))
    
    result = {"gamma_exp":gamma_exp_list,"C_exp":C_exp_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["gamma_exp","C_exp","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[12]:


def parameter_adjust_presion (X_train,y_train,X_val,y_val,sample_amount,data_ratio):
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    gamma_exp_list=[]
    C_exp_list=[]
    label_1_amount = int(sample_amount * (data_ratio/(data_ratio+1)))
    label_2_amount = int(sample_amount-label_1_amount)
    C_exp = -5
    while (C_exp <3):
        gamma_exp = -C_exp-12
        start1=time.time();
        train_data = pd.concat([X_train,y_train['label']],axis=1,join='outer')
        train_data = pd.DataFrame(train_data)
        Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
        Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],label_2_amount)
        res = [Class1_sample, Class2_sample]
        train_com = pd.concat(res)
        sample_label = pd.DataFrame(train_com['label'])
        sample_feature=train_com.drop(["label"],axis=1)

        svc_clf=SVC(kernel="rbf",degree=len(sample_feature),gamma=2**gamma_exp,C=2**C_exp)
        svc_clf.fit(sample_feature,sample_label['label'])
        y_pred=svc_clf.predict(X_val)
        y_pred = {"label_pred":y_pred}
        y_pred = pd.DataFrame(data=y_pred)
        #print ("curren gamma_exp:"+str(gamma_exp))
        #print ("current C_exp:"+str(C_exp))
        Precall,f1_score,BER,FPR= cal_score(y_pred,y_val['label'])
        temp=(time.time()-start1)/60
        time_list.append(temp)
        tpr_list.append(Precall)
        f1_score_list.append(f1_score)
        BER_list.append(BER)
        fpr_list.append(FPR)
        gamma_exp_list.append(gamma_exp)
        C_exp_list.append(C_exp)
        #print("fit time:%5.1fminute"%(temp))
    
    result = {"gamma_exp":gamma_exp_list,"C_exp":C_exp_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["gamma_exp","C_exp","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result

