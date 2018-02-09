
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
import itertools


# In[2]:


import sys
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from calculation_score import cal_score


# In[3]:


def nn_distance_calculate(X_val,X_train,y_train):
    class_1=X_train[y_train['label']==1]
    class_2=X_train[y_train['label']==2]
    #print(class_1.shape)
    #print(class_2.shape)
    mean_1=np.array(class_1.mean()).reshape(1,-1)
    mean_2=np.array(class_2.mean()).reshape(1,-1)
    #print(mean_2.shape)
    cov_1=np.matrix(class_1.cov())
    cov_2=np.matrix(class_2.cov())
    dis_1=distance.cdist(X_val,mean_1,metric='mahalanobis',V=cov_1).ravel()
    dis_2=distance.cdist(X_val,mean_2,metric='mahalanobis',V=cov_2).ravel()
    return dis_1, dis_2


# In[4]:


#Nearest Mean algorithm
def nn_predict(dis_1,dis_2,alpha,X_val):
    y_pred=[]
    for index in range(len(X_val)):
        if (alpha)*dis_1[index]<(1-alpha)*dis_2[index]:
                y_pred.append(np.float64(1.0)) 
        else:
                y_pred.append(np.float64(2.0))
    y_pred=pd.Series(y_pred)
    return y_pred


# In[7]:


def nn_predict_with_distance_adjust (training_feature,training_label):#,#X_val,y_val):
    alpha_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(training_feature,training_label['label'])
    alpha = 0.10
    
    while(alpha<=0.9):
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        for train_index, test_index in skf.split(training_feature,training_label['label']):
            X_train, X_val = training_feature.loc[train_index], training_feature.loc[test_index]
            y_train, y_val = training_label.loc[train_index], training_label.loc[test_index]

            dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)

            #print ("current alpha:"+str(alpha))
            y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
            Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)
        
        alpha_list.append(alpha)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        if (0.4<=alpha<=0.6):
            alpha=alpha+0.01
        else:
            alpha=alpha+0.1        
    result = {"alpha":alpha_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list}
    columns = ["alpha","f1_score","TPR","FPR","BER"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[8]:


def nn_predict_with_distance_adjust_presion (training_feature,training_label,alpha_lower_bound,alpha_higher_bound):
    #dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)
    alpha =alpha_lower_bound
    alpha_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(training_feature,training_label['label'])
    
    while(alpha<=alpha_higher_bound):
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        for train_index, test_index in skf.split(training_feature,training_label['label']):
            X_train, X_val = training_feature.loc[train_index], training_feature.loc[test_index]
            y_train, y_val = training_label.loc[train_index], training_label.loc[test_index]
            
            dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)
            y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
            Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])
            
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)

        alpha_list.append(alpha)
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        alpha=alpha+0.001
    
    result = {"alpha":alpha_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list}
    columns = ["alpha","f1_score","TPR","FPR","BER"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[15]:


def nn_validation (X_train,y_train,X_val,y_val,alpha):
    dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)
    y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
    Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])
    print ("TPR:"+str(Precall)+"   f1 score:" + str(f1_score)+"   FPR:"+ str(FPR)+"   BER:" + str(BER))
    return


# In[16]:


def nn_feature_selection_wrap(training_feature,training_label,alpha):
    feature_0_list = []
    feature_1_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    alpha_list=[]
    
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    skf.get_n_splits(training_feature,training_label['label'])
    
    feature_avaliable = ['feature0','feature1','feature2','feature3','feature4']
    feature_choice=list(itertools.combinations(feature_avaliable ,2))
   
    for i in range(len(feature_choice)):
        tpr_list_temp = []
        fpr_list_temp = []
        BER_list_temp = []
        f1_score_list_temp = []
        for train_index, test_index in skf.split(training_feature,training_label['label']):
            X_train, X_val = training_feature.loc[train_index], training_feature.loc[test_index]
            y_train, y_val = training_label.loc[train_index], training_label.loc[test_index]
            X_train=X_train.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
            X_val=X_val.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
            dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)
            y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
            Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])           
            tpr_list_temp.append(Precall)
            fpr_list_temp.append(FPR)
            BER_list_temp.append(BER)
            f1_score_list_temp.append(f1_score)

        alpha_list.append(alpha)
        feature_0_list.append(feature_choice[i][0])
        feature_1_list.append(feature_choice[i][1])
        tpr_list.append(sum(tpr_list_temp)/len(tpr_list_temp))
        fpr_list.append(sum(fpr_list_temp)/len(fpr_list_temp))
        BER_list.append(sum(BER_list_temp)/len(BER_list_temp))
        f1_score_list.append(sum(f1_score_list_temp)/len(f1_score_list_temp))
        alpha=alpha+0.001
    
    result = {"alpha":alpha_list,"feature_0":feature_0_list,"feature_1":feature_1_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list}
    columns = ["alpha","feature_0","feature_1","f1_score","TPR","FPR","BER"]
    result = pd.DataFrame (data=result,columns=columns)
    return result

