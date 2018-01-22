
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import re
import sys
sys.path.append(r"C:\Users\yangshanqi\Documents\lab\labgithubcopy\task_1_version_2")
from NN_function import nn_distance_calculate
from NN_function import nn_predict


# In[3]:


def nn_plot_prediction(stuck,slices,features,X_train,y_train,X_val,y_val,alpha):
    dataset = loadmat ("D:\\lab_signal_processing\\forStudents\\medData\\dataset.mat",mat_dtype=True)
    data=dataset["dataset"]
    dis_1, dis_2 =nn_distance_calculate(X_val,X_train,y_train)
    y_pred=nn_predict(dis_1,dis_2,alpha,X_val)
    y_pred = {"label_pred":y_pred}
    y_pred = pd.DataFrame(data=y_pred)
    plt.subplot(121)
    plt.title("original label")
    target_data=data[stuck][0][0][0][0][:,:,slices,features]
    target_image1=target_data.reshape(target_data.shape[0],target_data.shape[1])
    target_label=data[stuck][0][0][0][3][:,:,slices]
    target_image2=target_label.reshape(target_label.shape[0],target_label.shape[1])
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['black','blue','red'],256)
    plt.imshow(target_image1,cmap = matplotlib.cm.binary,alpha=0.6)
    plt.imshow(target_image2,cmap = cmap1,interpolation="bilinear",alpha=0.4)
    plt.axis("off")
    target_image=target_data.reshape(target_data.shape[0],target_data.shape[1])
    plt.show()
    
    selected_matrix=y_val[y_val['source_of_pixel']==stuck]
    selected_matrix=pd.DataFrame.reset_index(selected_matrix).drop('index',axis=1)
    temp_label = y_pred[y_val['source_of_pixel']==stuck]
    temp_label =pd.DataFrame.reset_index(temp_label ).drop('index',axis=1)
    cancer_x = []
    cancer_y = []
    noncancer_x = []
    noncancer_y = []
    for i in range(len(selected_matrix)):
            temparr = re.findall("\d+",selected_matrix.iloc[i]['index_of_pixel'])
            if (int(temparr[2]) == slices):
                if (temp_label.iloc[i]['label_pred']==2):
                    cancer_x.append(int(temparr[0]))
                    cancer_y.append(int(temparr[1]))
                else:
                    noncancer_x.append(int(temparr[0]))
                    noncancer_y.append(int(temparr[1]))
    plt.subplot(122)
    plt.title("prediction with NN")
    plt.imshow(target_image1,cmap = matplotlib.cm.binary,alpha=0.8)
    plt.scatter(noncancer_y,noncancer_x,label="predict_non_cancer",s = 3)
    plt.scatter(cancer_y,cancer_x,label="predict_cancer",s = 2)
    plt.legend()
    plt.axis("off")
    plt.show()

