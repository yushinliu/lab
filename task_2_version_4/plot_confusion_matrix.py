import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
#default

import pickle

save_path="D:\\LAB\\lab\\task_2_version_4\\confusion_matrix.txt"
#save_path="/Users/Mata/Documents/lab/task_2_version_3/features.txt"
f = open(save_path,'rb')
confusion_matrix=pickle.load(f)
f.close()

print(confusion_matrix.shape)
num_samples=confusion_matrix.shape[0]
conf_mat=pd.DataFrame(confusion_matrix)#,index=[i for i in range(num_samples)],columns=[i for i in range(num_samples)])
plt.figure(figsize=(18,10))
sn.heatmap(conf_mat,annot=True)
plt.show()