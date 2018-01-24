
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
import sys
import itertools
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from calculation_score import cal_score


# In[2]:


def kNN_k_parameter_adjust (X_train,y_train,X_val,y_val):
    start=time.time()
    k_value=5
    K_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    while (k_value<300):
        start1=time.time()
        neigh = KNeighborsClassifier(n_neighbors=k_value,algorithm='auto',weights ='distance')
        neigh.fit(X_train, y_train['label'])
        y_pred = neigh.predict(X_val)
        y_pred = {"label_pred":y_pred}
        y_pred = pd.DataFrame(data=y_pred)
        Precall,f1_score,BER,FPR = cal_score (y_pred,y_val['label']) 
        K_list.append(k_value)
        tpr_list.append(Precall)
        fpr_list.append(FPR)
        BER_list.append(BER)
        f1_score_list.append(f1_score)
        temp=(time.time()-start1)/60
        time_list.append(temp)
        #print("current k_value:%d ,fit time:%5.1fminute"%(k_value,(time.time()-start1)/60))
        if (k_value<=20):
            k_value=k_value+1
        else:
            k_value=k_value+3
        #print("                 ")
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))
    result = {"k_value":K_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["k_value","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[6]:


def kNN_data_ratio_adjust (X_train,y_train,X_val,y_val,k_value):
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
    while label_1_amount > 9000:
        count=0
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]
        while (count<3):
            train_data = {'feature0':X_train['feature0'],'feature1':X_train['feature1'],'feature2':X_train['feature2'],'feature3':X_train['feature3'],'feature4':X_train['feature4'],'label':y_train['label']}
            train_data = pd.DataFrame(data=train_data)
            Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
            Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],20000)
            res = [Class1_sample, Class2_sample]
            train_com = pd.concat(res)
            #print(train_com)
            sample_label = pd.DataFrame(train_com['label'])
            sample_feature=train_com.drop(["label"],axis=1)
            start1=time.time()

            neigh = KNeighborsClassifier(n_neighbors=k_value,algorithm='auto',weights ='distance')
            neigh.fit(sample_feature, sample_label['label'])
            y_pred = neigh.predict(X_val)
            y_pred = {"label_pred":y_pred}
            y_pred = pd.DataFrame(data=y_pred)

            Precall,f1_score,BER,FPR = cal_score (y_pred,y_val['label'])           
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            temp=(time.time()-start1)/60
            time_list_temp.append(temp)
            count=count+1
        
        label_1_amount_list.append(label_1_amount)
        label_2_amount_list.append(20000)
        ratio_list.append(label_1_amount/20000)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        #print("current data labe 1 size:%d ,fit time:%5.1fminute"%(t,(time.time()-start1)/60))
        label_1_amount = label_1_amount-2000
    
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))       
    result = {"label_1_amount":label_1_amount_list,"label_2_amount":label_2_amount_list,"label 1: label 2 ratio":ratio_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["label_1_amount","label_2_amount","label 1: label 2 ratio","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[7]:


def kNN_data_ratio_adjust_2 (X_train,y_train,X_val,y_val,k_value):
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
    while label_1_amount > 3000:
        count=0
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]
        while (count<3):
            train_data = {'feature0':X_train['feature0'],'feature1':X_train['feature1'],'feature2':X_train['feature2'],'feature3':X_train['feature3'],'feature4':X_train['feature4'],'label':y_train['label']}
            train_data = pd.DataFrame(data=train_data)
            Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
            Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],8000)
            res = [Class1_sample, Class2_sample]
            train_com = pd.concat(res)
            #print(train_com)
            sample_label = pd.DataFrame(train_com['label'])
            sample_feature=train_com.drop(["label"],axis=1)
            start1=time.time()

            neigh = KNeighborsClassifier(n_neighbors=k_value,algorithm='auto',weights ='distance')
            neigh.fit(sample_feature, sample_label['label'])
            y_pred = neigh.predict(X_val)
            y_pred = {"label_pred":y_pred}
            y_pred = pd.DataFrame(data=y_pred)

            Precall,f1_score,BER,FPR = cal_score (y_pred,y_val['label'])           
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            temp=(time.time()-start1)/60
            time_list_temp.append(temp)
            count=count+1
        
        label_1_amount_list.append(label_1_amount)
        label_2_amount_list.append(8000)
        ratio_list.append(label_1_amount/8000)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        #print("current data labe 1 size:%d ,fit time:%5.1fminute"%(t,(time.time()-start1)/60))
        label_1_amount = label_1_amount-6000
    
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))       
    result = {"label_1_amount":label_1_amount_list,"label_2_amount":label_2_amount_list,"label 1: label 2 ratio":ratio_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["label_1_amount","label_2_amount","label 1: label 2 ratio","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[8]:


def kNN_k_parameter_adjust_with_bisaes_data (X_train,y_train,X_val,y_val,data_ratio):
    start=time.time()
    k_value=5
    K_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    time_list=[]
    train_data = {'feature0':X_train['feature0'],'feature1':X_train['feature1'],'feature2':X_train['feature2'],'feature3':X_train['feature3'],'feature4':X_train['feature4'],'label':y_train['label']}
    train_data = pd.DataFrame(data=train_data)
    Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],int(8000*data_ratio))
    Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],8000)
    res = [Class1_sample, Class2_sample]
    train_com = pd.concat(res)
    #print(train_com)
    sample_label = pd.DataFrame(train_com['label'])
    sample_feature=train_com.drop(["label"],axis=1)
   
    while (k_value<300):
        
        start1=time.time()
        neigh = KNeighborsClassifier(n_neighbors=k_value,algorithm='auto',weights ='distance')
        neigh.fit(sample_feature, sample_label['label'])
        y_pred = neigh.predict(X_val)
        y_pred = {"label_pred":y_pred}
        y_pred = pd.DataFrame(data=y_pred)
        Precall,f1_score,BER,FPR = cal_score (y_pred,y_val['label']) 
        K_list.append(k_value)
        tpr_list.append(Precall)
        fpr_list.append(FPR)
        BER_list.append(BER)
        f1_score_list.append(f1_score)
        temp=(time.time()-start1)/60
        time_list.append(temp)
        #print("current k_value:%d ,fit time:%5.1fminute"%(k_value,(time.time()-start1)/60))
        if (k_value<=20):
            k_value=k_value+1
        else:
            k_value=k_value+3
        #print("                 ")
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))
    result = {"k_value":K_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["k_value","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[12]:


def best_result_choose(X_train_temp,y_train,X_val_temp,y_val,k_value):
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
    while label_1_amount > 9000:
        count=0
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        time_list_temp=[]
        while (count<2):
            train_data = pd.concat([X_train_temp,y_train['label']],axis=1,join='outer')
            train_data = pd.DataFrame(data=train_data)
            Class1_sample =pd.DataFrame.sample(train_data[train_data['label']==1],label_1_amount)
            Class2_sample =pd.DataFrame.sample(train_data[train_data['label']==2],20000)
            res = [Class1_sample, Class2_sample]
            train_com = pd.concat(res)
            #print(train_com)
            sample_label = pd.DataFrame(train_com['label'])
            sample_feature=train_com.drop(["label"],axis=1)
            #print(sample_label)
            #print(sample_feature)
            start1=time.time()

            neigh = KNeighborsClassifier(n_neighbors=k_value,algorithm='auto',weights ='distance')
            neigh.fit(sample_feature, sample_label['label'])
            y_pred = neigh.predict(X_val_temp)
            y_pred = {"label_pred":y_pred}
            y_pred = pd.DataFrame(data=y_pred)

            Precall,f1_score,BER,FPR = cal_score (y_pred,y_val['label'])           
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
            temp=(time.time()-start1)/60
            time_list_temp.append(temp)
            count=count+1
        
        label_1_amount_list.append(label_1_amount)
        label_2_amount_list.append(20000)
        ratio_list.append(label_1_amount/20000)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        time_list.append(sum(time_list_temp)/len(time_list_temp))
        #print("current data labe 1 size:%d ,fit time:%5.1fminute"%(t,(time.time()-start1)/60))
        label_1_amount = label_1_amount-5000
    
    print("the total executing time:%5.1fminute"%((time.time()-start)/60))       
    result = {"label_1_amount":label_1_amount_list,"label_2_amount":label_2_amount_list,"label 1: label 2 ratio":ratio_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list,"time":time_list}
    columns = ["label_1_amount","label_2_amount","label 1: label 2 ratio","f1_score","TPR","FPR","BER","time"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[10]:


def kNN_feature_selection_wrap(X_train,y_train,X_val,y_val,k_value):
    feature_avaliable = ['feature0','feature1','feature2','feature3','feature4']
    feature_choice=list(itertools.combinations(feature_avaliable ,2))
    for i in range(len(feature_choice)):
        print("current feature choosen:")
        print(feature_choice[i])
        print("                         ")
        X_train_temp = X_train.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
        X_val_temp = X_val.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
        result = best_result_choose (X_train_temp,y_train,X_val_temp,y_val,k_value)
        print(result.loc[result['f1_score'].idxmax()])
        print("              ")

