import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

dataset=pd.read_csv('D:/5thsem/DWM/Crop_recommendation (1).csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

print(x_test)