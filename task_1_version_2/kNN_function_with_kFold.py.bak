
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import sys
import itertools
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from calculation_score import cal_score


# In[2]:


def kNN_base_function (X_train,y_train,X_val,y_val,k_value):
    neigh = KNeighborsClassifier(n_neighbors=k_value,algorithm='auto',weights ='distance')
    neigh.fit(X_train, y_train['label'])
    y_pred = neigh.predict(X_val)
    y_pred = {"label_pred":y_pred}
    y_pred = pd.DataFrame(data=y_pred)
    Precall,f1_score,BER,FPR = cal_score (y_pred,y_val['label']) 
    return Precall,f1_score,BER,FPR


# In[37]:


def kNN_k_parameter_adjust (training_feature,training_label):
    start=time.time()
    k_value=5
    K_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(training_feature,training_label['label'])
    
    while (k_value<300):
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp = []
        for train_index, test_index in skf.split(training_feature,training_label['label']):
            start1=time.time()
            X_train, X_val = training_feature.loc[train_index], training_feature.loc[test_index]
            y_train, y_val = training_label.loc[train_index], training_label.loc[test_index]

            Precall,f1_score,BER,FPR = kNN_base_function (X_train,y_train,X_val,y_val,k_value)

            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            time_list_temp.append(time.time()-start1)
        
        K_list.append(k_value)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        k_value= k_value+3
        
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))
    result = {"k_value":K_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["k_value","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[39]:


def kNN_data_ratio_adjust (training_feature,training_label,k_value):
    start=time.time()
    label_1_amount = 40000
    label_1_amount_list = []
    label_2_amount_list = []
    ratio_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(training_feature,training_label['label'])
    
    while (label_1_amount > 2000):
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]

        for train_index, test_index in skf.split(training_feature,training_label['label']):
            start1=time.time()
            X_train, X_val = training_feature.loc[train_index], training_feature.loc[test_index]
            y_train, y_val = training_label.loc[train_index], training_label.loc[test_index]
 
            train_data = pd.concat([X_train,y_train['label']],axis=1,join='outer')
            Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
            Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],10000)
            res = [Class1_sample, Class2_sample]
            train_com = pd.concat(res)
            y_train = pd.DataFrame(train_com['label'])
            X_train=train_com.drop(["label"],axis=1)
            
            Precall,f1_score,BER,FPR = kNN_base_function (X_train,y_train,X_val,y_val,k_value)  
            
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            time_list_temp.append(time.time()-start1) 
            
        label_1_amount_list.append(label_1_amount)
        label_2_amount_list.append(10000)
        ratio_list.append(label_1_amount/10000)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        #print("current data labe 1 size:%d ,fit time:%5.1fminute"%(t,(time.time()-start1)/60))
        if (label_1_amount > 10000):
            label_1_amount = label_1_amount-5000
        else:
            label_1_amount = label_1_amount-2500
        
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))       
    result = {"label_1_amount":label_1_amount_list,"label_2_amount":label_2_amount_list,"label 1: label 2 ratio":ratio_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["label_1_amount","label_2_amount","label 1: label 2 ratio","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result
            
            


# In[1]:


def kNN_k_parameter_adjust_with_specific_data_ratio (X_train,y_train,data_ratio):
    
    train_data = pd.concat(X_train,y_train['label'],1)
    train_data = pd.DataFrame(data=train_data)
    Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],int(8000*data_ratio))
    Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],8000)
    res = [Class1_sample, Class2_sample]
    train_com = pd.concat(res)
    #print(train_com)
    sample_label = pd.DataFrame(train_com['label'])
    sample_feature=train_com.drop(["label"],axis=1)
   
    result = kNN_k_parameter_adjust (training_feature,training_label)
    
    return result


# In[3]:


def kNN_feature_selection_wrap(X_train,y_train,k_value):
    feature_avaliable = ['feature0','feature1','feature2','feature3','feature4']
    feature_choice=list(itertools.combinations(feature_avaliable ,2))
    for i in range(len(feature_choice)):
        count = 0
        print("current feature choosen:")
        print(feature_choice[i])
        print("                         ")
        X_train_temp = X_train.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
        X_val_temp = X_val.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
        
        res = kNN_data_ratio_adjust (X_train_temp,X_val_temp ,k_value)
        res['feature0']=feature_choice[i][0]
        res['feature1']=feature_choice[i][1]
        if (count==0):
            res_total = res
            count = count+1
        else :
            res_temp =[res_total,res]
            res_total = pd.concat (res_temp,0)
    return res_total
        #print("              ")

