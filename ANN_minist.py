# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:57:21 2018

@author: Zehao Huang
"""
import matplotlib.pyplot as plt
import numpy as np
import struct
from set import create_layers,positive_calculation,get_delta,update_weights,calculate_testoutput,reset_alpha
import time

time_start=time.time()
def read_image_files(filename, num):
    bin_file = open(filename, 'rb')
    buf = bin_file.read()
    index = 0
    # 前四个32位integer为以下参数
    # >IIII 表示使用大端法读取
    magic, numImage, numRows, numCols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    image_sets = []
    for i in range(num):
        images = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        images = np.array(images)
        images = images/255.0
        images = images.tolist()
        # if i == 6:
        #     print ','.join(['%s'%x for x in images])
        image_sets.append(images)
    bin_file.close()
    return image_sets


def read_label_files(filename):
    bin_file = open(filename, 'rb')
    buf = bin_file.read()
    index = 0
    magic, nums = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labels = struct.unpack_from('>%sB'%nums, buf, index)
    bin_file.close()
    labels = np.array(labels)
    return labels


def fetch_traingset():
    image_file = 'train-images.idx3-ubyte'
    label_file = 'train-labels.idx1-ubyte'
    images = read_image_files(image_file,60000)
    labels = read_label_files(label_file)
    return {'images': images,
            'labels': labels}


def fetch_testingset():
    image_file = 't10k-images.idx3-ubyte'
    label_file = 't10k-labels.idx1-ubyte'
    images = read_image_files(image_file,10000)
    labels = read_label_files(label_file)
    return {'images': images,
            'labels': labels}
    
def get_sets(start,end,Type):
    setdata=np.zeros([end-start,784])
    getlabels=np.zeros([end-start,1])
    if Type == 'train':
        setdata=fetch_traingset()['images'][start: end]
        setdata=np.array(setdata) 
        getlabels=fetch_traingset()['labels'][start: end]
        setlabels=np.zeros([end-start,10])
        a=0
        for i in getlabels:
            setlabels[a][i]=1
            a=a+1
        setlabels=np.array(setlabels)
    else:
        setdata=fetch_testingset()['images'][start: end]
        setdata=np.array(setdata) 
        getlabels=fetch_testingset()['labels'][start: end] 
        setlabels=np.zeros([end-start,10])
        a=0
        for i in getlabels:
            setlabels[a][i]=1
            a=a+1
        setlabels=np.array(setlabels)
    mean = setdata.mean(axis=0)
    std = setdata.std(axis=0)
    setdata = (setdata - mean) / (std+1e-10)
    return setdata,setlabels

def get_accuracy(setlabels,resultlabels):
    Loc=np.argmax(setlabels, axis=1)
    a=0
    error=0
    for i in Loc:
        if resultlabels[a][i]<0.5:
            error=error+1
        a=a+1
    labelnums=setlabels.shape[0]
    accuracy=(labelnums-error)/labelnums
    return accuracy

'''def OvM(traindata,trainlabels,valdata,vallabels,testdata,testlabels):
    for c in range(10):
        layers=create_layers([0.0005,0.0005,0.0005,0.0005],['RELU','RELU','RELU','Sigmoid'],[(784,1024),(1024,512),(512,256),(256,1)],4)
        layers[0].setoutput(traindata)
        layers[0].settestoutput(testdata[1:10000][c])
        resultlabels=np.zeros([10000,10])
        for j in range(10):
            layers=positive_calculation(layers)
            layers= calculate_testoutput(layers)
            resultlabels[1:10000][c]=layers[-1].testoutput
            print('.')
    accuracy=get_accuracy(testlabels,resultlabels)
    print('accuracy:',accuracy)
    return resultlabels'''
        
traindata,trainlabels=get_sets(0,50000,'train')
valdata,vallabels=get_sets(50000,60000,'train')
testdata,testlabels=get_sets(0,10000,'test')

#                     Neural        Network                     #
Learning_Rate=0.001
layers=create_layers([Learning_Rate,Learning_Rate,Learning_Rate,Learning_Rate,Learning_Rate],['RELU','RELU','RELU','RELU','Sigmoid'],[(784,1024),(1024,512),(512,256),(256,128),(128,10)],5)

layers[0].setoutput(traindata)
layers[0].settestoutput(testdata)

valac=np.zeros(20000)  
for j in range(5000):
    print('iteration: ',j+1,'/1000')
    layers=positive_calculation(layers)
    layers=get_delta(layers,trainlabels)
    layers=update_weights(layers)
    
    layers[0].settestoutput(valdata)
    layers= calculate_testoutput(layers)
    valoutput=layers[-1].testoutput
    valac[j]=get_accuracy(vallabels,valoutput)
    print('valac:',valac)
    
    layers[0].settestoutput(testdata)
    layers= calculate_testoutput(layers)
    resultlabels=layers[-1].testoutput
    accuracy=get_accuracy(testlabels,resultlabels)
    print('accuracy:',accuracy)
    cost=np.sum(np.square(testlabels - resultlabels))/2
    print('iteration: ',j+1,'/1000','  ','val_ac:',valac[j],'   ','loss:',cost)
    if j>50&j<100:  
        layers=reset_alpha(layers,0.0005)
    if j>100 :
        layers=reset_alpha(layers,0.00025)
    if j>200 :
        layers=reset_alpha(layers,0.0001)
    if j>350:
        layers=reset_alpha(layers,0.00001)

for i in range(20000):
    if i>=500:
        valac[i]=valac[499]
        
plt.figure(1)
plt.plot(valac, 'g', label='validation_acc')
plt.legend()

time_end=time.time()
print('time cost',time_end-time_start,'s')