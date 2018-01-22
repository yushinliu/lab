
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.spatial import distance
import itertools


# In[5]:


import sys
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from calculation_score import cal_score


# In[2]:


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


# In[3]:


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


# In[4]:


def nn_predict_with_distance_adjust (X_train,y_train,X_val,y_val):
    dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)
    alpha = 0.10
    alpha_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    while(alpha<=0.9):
        #print ("current alpha:"+str(alpha))
        y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
        Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])
        alpha_list.append(alpha)
        tpr_list.append(Precall)
        fpr_list.append(FPR)
        BER_list.append(BER)
        f1_score_list.append(f1_score)
        if (0.4<=alpha<=0.6):
            alpha=alpha+0.01
        else:
            alpha=alpha+0.1
        #print("                 ")
    result = {"alpha":alpha_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list}
    columns = ["alpha","f1_score","TPR","FPR","BER"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[6]:


def nn_predict_with_distance_adjust_presion (X_train,y_train,X_val,y_val,alpha_lower_bound,alpha_higher_bound):
    dis_1,dis_2=nn_distance_calculate(X_val,X_train,y_train)
    alpha =alpha_lower_bound
    alpha_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    while(alpha<=alpha_higher_bound):
        #print ("current alpha:"+str(alpha))
        y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
        Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])
        alpha_list.append(alpha)
        tpr_list.append(Precall)
        fpr_list.append(FPR)
        BER_list.append(BER)
        f1_score_list.append(f1_score)
        alpha=alpha+0.001
    result = {"alpha":alpha_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list}
    columns = ["alpha","f1_score","TPR","FPR","BER"]
    result = pd.DataFrame (data=result,columns=columns)
    return result


# In[7]:


def best_result_choosen (dis_1,dis_2,X_val,y_val):
    alpha = 0.47
    alpha_list = []
    tpr_list = []
    fpr_list = []
    BER_list = []
    f1_score_list = []
    while(alpha<=0.57):
        #print ("current alpha:"+str(alpha))
        y_pred_temp=nn_predict(dis_1,dis_2,alpha,X_val)
        Precall,f1_score,BER,FPR = cal_score (y_pred_temp,y_val['label'])
        alpha_list.append(alpha)
        tpr_list.append(Precall)
        fpr_list.append(FPR)
        BER_list.append(BER)
        f1_score_list.append(f1_score)
        alpha=alpha+0.002
    
    max_f1_score_index=f1_score_list.index(max(f1_score_list))
    result = {"alpha":alpha_list,"TPR":tpr_list,"FPR":fpr_list,"f1_score":f1_score_list,"BER":BER_list}
    columns = ["alpha","f1_score","TPR","FPR","BER"]
    result = pd.DataFrame (data=result,columns=columns)
    print(result.loc[result['f1_score'].idxmax()])
    return 0
    


# In[8]:


def nn_feature_selection_wrap(X_train,y_train,X_val,y_val):
    feature_avaliable = ['feature0','feature1','feature2','feature3','feature4']
    feature_choice=list(itertools.combinations(feature_avaliable ,2))
    for i in range(len(feature_choice)):
        print("current feature choosen:")
        print(feature_choice[i])
        print("                         ")
        X_train_temp = X_train.loc[:,[feature_choice[i][0],feature_choice[i][1]]]
        X_val_temp = X_val.loc[:,[feature_choice[i][0],feature_choice[i][1]]]                      
        dis_1,dis_2 = nn_distance_calculate(X_val_temp,X_train_temp,y_train)
        best_result_choosen (dis_1,dis_2,X_val,y_val)
        print("              ")

