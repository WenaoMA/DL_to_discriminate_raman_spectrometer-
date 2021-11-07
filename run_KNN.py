# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt

import math
import random
import time

import scipy.io as scio
import tensorflow.keras.backend as K
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from ROC2_PCA import evaluation


data = np.load('./data/PCA_FrozenTumor_cut0.npy')
save_path = './result_save2/KNN_FrozenTumor_cut0/'
data=data.transpose() 
data=pd.DataFrame(data)
data.tail()


time_start = time.time()

stock=data
for Test_num in range(10):


    train_test_rate = 9.0/10
    
    

    amount_of_features = len(stock.columns)
    data = stock.values 

    result = []
    result_y = []
    for j in range(data.shape[0]):
        result.append(data[j,:-1])
        result_y.append(data[j,-1])
    result = np.array(result)
    row = round((1-train_test_rate) * result.shape[0])
    
    
    x_test = result[Test_num*int(row):Test_num*int(row)+int(row),:]
    y_test = result_y[Test_num*int(row):Test_num*int(row)+int(row):]
    
    
    
    
    num_ = list(range(0,int(train_test_rate * result.shape[0])))
    random.shuffle(num_)
    train = np.zeros((int(result.shape[0]*train_test_rate),result.shape[1])).astype(np.float32)
    train_label = np.zeros((int(result.shape[0]*train_test_rate))).astype(np.float32)
    j = 0
    for i in range(int(result.shape[0])):
        if i < Test_num*int(row) or i >= Test_num*int(row)+int(row):
            train[j,:] = result[i,:]
            train_label[j] = result_y[i]
            j += 1

    
    x_train = np.zeros((int(result.shape[0]*train_test_rate),result.shape[1])).astype(np.float32)
    y_train = np.zeros((int(result.shape[0]*train_test_rate))).astype(np.float32)
    N = 0
    for z in num_:
        x_train[N,:] = train[z,:]
        y_train[N] = train_label[z]
        N += 1

    
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))  
    
    pos = np.where(y_train==1)
    num_pos = len(pos[0])
    neg = np.where(y_train==0)
    num_neg = len(neg[0])
    pos_ratio = num_pos / (num_pos+num_neg)
    neg_ratio = num_neg / (num_pos+num_neg)
    
    X_train.shape,X_test.shape
    y_test = np.array(y_test)
    
    
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)

    X_test_PCA = pca.transform(X_test)
    
    
    knn = KNeighborsClassifier()
    knn.fit(X_train_PCA, y_train)
    test_predict_label  = knn.predict_proba(X_test_PCA)

    y_test_predict=knn.predict(X_test_PCA)

    from sklearn import metrics
    print("精确度等指标：")
    print(metrics.classification_report(y_test,y_test_predict))
    print("混淆矩阵：")
    print(metrics.confusion_matrix(y_test,y_test_predict))
    last_result = metrics.confusion_matrix(y_test,y_test_predict)

    np.save(save_path+str(Test_num)+'_result.npy',last_result)
    np.save(save_path+str(Test_num)+'_probabilityMap.npy',test_predict_label[:,1])
    np.save(save_path+str(Test_num)+'_GT.npy',y_test)

evaluation(save_path)