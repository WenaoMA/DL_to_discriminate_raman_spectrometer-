# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

import math
import random
import time
import scipy.io as scio
from functools import partial
from sklearn import svm
from ROC2_PCA import evaluation

# load data
window=1015
data = np.load('./data/PCA_NormalTumor_cut0.npy')
save_path = './result_save2/SVM_NormalTumor_cut0/'
data=data.transpose() 
data=pd.DataFrame(data)
data.tail()
tumor_dic = {'G27T_start':345,'G27T_end':364,'G28T_start':365,'G28T_end':384,
'G29T_start':385,'G29T_end':401,'G31T_start':402,'G31T_end':420,
'G32T_start':421,'G32T_end':439,'G33T_start':440,'G33T_end':459,
'G34T_start':460,'G34T_end':477,'G35T_start':478,'G35T_end':495,
'G36T_start':496,'G36T_end':515,'G37T_start':516,'G37T_end':532,
'G39T_start':533,'G39T_end':551,'G41T_start':552,'G41T_end':571,
'G42T_start':572,'G42T_end':590,'G46T_start':591,'G46T_end':608,
'G51T_start':609,'G51T_end':626,'G52T_start':627,'G52T_end':646,
'G54T_start':647,'G54T_end':665,'G56T_start':666,'G56T_end':681,
'G57T_start':682,'G57T_end':701,'G59T_start':702,'G59T_end':720,
'G60T_start':721,'G60T_end':740,'G63T_start':741,'G63T_end':760,
'G64T_start':761,'G64T_end':777,'G65T_start':778,'G65T_end':796,
'G67T_start':797,'G67T_end':816,'G68T_start':817,'G68T_end':834,
'G69T_start':835,'G69T_end':854,'G70T_start':855,'G70T_end':873,
'G71T_start':874,'G71T_end':893,'G72T_start':894,'G72T_end':913,
'G73T_start':914,'G73T_end':932,'G74T_start':933,'G74T_end':951,
'G75T_start':952,'G75T_end':971,'G77T_start':972,'G77T_end':989,
'G78T_start':990,'G78T_end':1009,'G79T_start':1010,'G79T_end':1028,
'G80T_start':1029,'G80T_end':1048,'G82T_start':1049,'G82T_end':1065,
'G83T_start':1066,'G83T_end':1083,'G84T_start':1084,'G84T_end':1102,
'G85T2_start':1103,'G85T2_end':1120,'G86T_start':1121,'G86T_end':1137}


normal_dic = {'G53N_start':1,'G53N_end':20,'G54N_start':21,'G54N_end':40,'G56N_start':41,'G56N_end':59,
'G57N_start':60,'G57N_end':79,'G59N_start':80,'G59N_end':99,'G61N_start':100,'G61N_end':119,
'G66N_start':120,'G66N_end':139,'G67N_start':140,'G67N_end':159,'G69N_start':160,'G69N_end':179,
'G71N_start':180,'G71N_end':199,'G72N_start':200,'G72N_end':218,'G73N_start':219,'G73N_end':229,
'G78N_start':230,'G78N_end':248,'G83N_start':249,'G83N_end':268,'G84N_start':269,'G84N_end':286,
'G85N1_start':287,'G85N1_end':306,'G85N2_start':307,'G85N2_end':324,'G86N_start':325,'G86N_end':344}
stock=data

for crossEntropyNum in range(10):

    tumor_dic_key = sorted(tumor_dic)
    normal_dic_key = sorted(normal_dic)
    if crossEntropyNum<8:
        selectTumorNumber = 4
        selectNormalNumber = 2
    else:
        selectTumorNumber = 5
        selectNormalNumber = 1
    delNum = []
    
    
    time_start = time.time()
    
    Test_num = crossEntropyNum
    # train_test_rate = 9.0/10
    
    seq_len=window 
    # input_size=len(data.iloc[1,:])
    amount_of_features = len(stock.columns)
    data = stock.values  
    sequence_length = seq_len
    result = []
    result_y = []
    for j in range(data.shape[0]):
        result.append(data[j,:-1])
        result_y.append(data[j,-1])
    result = np.array(result)

    
    for i in range(0,selectTumorNumber*2,2):
      if crossEntropyNum ==9:
          start_num = tumor_dic[tumor_dic_key[74+i+1]]
          end_num = tumor_dic[tumor_dic_key[74+i]]
      elif crossEntropyNum ==8:
          start_num = tumor_dic[tumor_dic_key[64+i+1]]
          end_num = tumor_dic[tumor_dic_key[64+i]]
      else:
          start_num = tumor_dic[tumor_dic_key[crossEntropyNum*selectTumorNumber*2+i+1]]
          end_num = tumor_dic[tumor_dic_key[crossEntropyNum*selectTumorNumber*2+i]]
      delNum_ = list(range(start_num-1,end_num))
      delNum = delNum + delNum_
      if i ==0:
        x_test = result[start_num-1:end_num,:]
        y_test = result_y[start_num-1:end_num]
      else:
        x_test = np.concatenate((x_test,result[start_num-1:end_num,:]),axis=0)
        y_test = y_test + result_y[start_num-1:end_num]
    
    for i in range(0,selectNormalNumber*2,2):
      if crossEntropyNum ==9:
          start_num = normal_dic[normal_dic_key[34+i+1]]
          end_num = normal_dic[normal_dic_key[34+i]]  
      elif crossEntropyNum ==8:
          start_num = normal_dic[normal_dic_key[32+i+1]]
          end_num = normal_dic[normal_dic_key[32+i]]  
      else:
          start_num = normal_dic[normal_dic_key[crossEntropyNum*selectNormalNumber*2+i+1]]
          end_num = normal_dic[normal_dic_key[crossEntropyNum*selectNormalNumber*2+i]]
      delNum_ = list(range(start_num-1,end_num))
      delNum = delNum + delNum_
      x_test = np.concatenate((x_test,result[start_num-1:end_num,:]),axis=0)
      y_test = y_test + result_y[start_num-1:end_num]
    
    result_ = np.delete(result,delNum,axis=0)
    result_y_ = np.delete(result_y,delNum)
    
    
    num_ = list(range(0,result_.shape[0]))
    random.shuffle(num_)
    #cut_ = np.zeros((num,data.shape[1])).astype(np.float32)
    x_train = np.zeros((result_.shape[0],result_.shape[1])).astype(np.float32)
    y_train = np.zeros(len(result_y_)).astype(np.float32)
    N = 0
    for z in num_:
        x_train[N,:] = result_[z,:]
        y_train[N] = result_y_[z]
        N += 1
    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))  
    
    
    X_train.shape,X_test.shape
    y_test = np.array(y_test)
    
        
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)
    # pca.fit(X_test)
    X_test_PCA = pca.transform(X_test)
    
    
    clf = svm.SVC(kernel='rbf', C = 10)
    clf.fit(X_train_PCA, y_train)

    y_test_predict=clf.predict(X_test_PCA)
    test_predict_label  = clf.decision_function(X_test_PCA)
    fpr,tpr,threshold = roc_curve(y_test, test_predict_label) 

    roc_auc = auc(fpr,tpr)
    from sklearn import metrics
    print("evaluation matrix：")
    print(metrics.classification_report(y_test,y_test_predict))
    print("confusion matrix：")
    print(metrics.confusion_matrix(y_test,y_test_predict))
    last_result = metrics.confusion_matrix(y_test,y_test_predict)
    np.save(save_path+str(Test_num)+'_result.npy',last_result)
    np.save(save_path+str(Test_num)+'_probabilityMap.npy',test_predict_label)
    np.save(save_path+str(Test_num)+'_GT.npy',y_test)

evaluation(save_path)


