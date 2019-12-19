# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:14:46 2019

@author: Aniket Kambli
"""


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pandas_datareader as pdr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

def getdata(stockname,start_date,end_date):
    data=pdr.data.get_data_yahoo(stockname,start_date,end_date)
    return data
 

def split_data(data):
    data['Moving Average'] = data['Close'].rolling(window=18).mean()
    data=data.dropna()
    
    data['High/Low']=0
    
    for i in range(1,data.shape[0]):
            if(data['Close'][i] < data['Close'][i-1]):
                data['High/Low'][i]=1
            else:
                data['High/Low'][i]=0 
            

    length_of_data=data.shape[0]
    length80=int((length_of_data/100)*80)
            
    data=data.dropna()
    X=data[:length_of_data]
    y=data[:length_of_data]
    
    X=data[['High','Close','Adj Close','Moving Average']]
    y=data['High/Low']
    
    X.reset_index(drop=True,inplace=True)
    y.reset_index(drop=True,inplace=True)
    
    X_train=X.iloc[:length80]
    X_test=X.iloc[length80:]
    
    y_train=y.iloc[:length80]
    y_test=y.iloc[length80:]
    return X_train,X_test,y_train,y_test

def model_logistic_regression(X_train,X_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    from sklearn.preprocessing import StandardScaler
    scx=StandardScaler()
    X_train=scx.fit_transform(X_train)
    X_test=scx.transform(X_test)
    model.fit(X_train,y_train)
    ypreds=model.predict(X_test)
    print(classification_report(y_test,ypreds))
    print(confusion_matrix(y_test,ypreds))
    


data=pdr.data.get_data_yahoo('Reliance.Ns','2015-01-01','2019-01-22') 
X_train,X_test,y_train,y_test=split_data(data)
model_logistic_regression(X_train,X_test,y_train,y_test)

