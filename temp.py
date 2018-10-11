# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class layer(object):
    def __init__(self,alpha,activation,size):
        self.activation=activation   #激活函数类型Sigmoid，RELU，tanh 
        self.size=size               #大小
        self.weights=2*np.random.random(size)-1   #初始化权重
        self.alpha=alpha            #学习率
        self.upperlayer=None
        self.lowerlayer=None
        self.b=np.random.random([1,size[1]])
    
    def positive(self):  #正向计算
        self.output=activation(self.activation,np.dot(self.lowerlayer.output,self.weights)-self.b)
    def getdelta(self,y_train):    #计算delta值
        if self.upperlayer == None:  #最高层（上面是输出）
            self.delta=(y_train - self.output)*derivation(self.activation,self.output)
        else:
            self.delta=self.upperlayer.delta.dot(self.upperlayer.weights.T)*derivation(self.activation,self.output)
    def update(self):  #更新权
        self.weights += self.alpha * self.lowerlayer.output.T.dot(self.delta)
        #print('sizeofweight:',np.shape(self.weights))
        self.b -=self.alpha*self.delta.mean()
    def setoutput(self,output):  #设置输出（用于设置输入层）
        self.output= output
    def setlayers(self,lowerlayer,upperlayer): #连接各层之间的关系
        self.lowerlayer = lowerlayer
        self.upperlayer= upperlayer
    def settestoutput(self,X_test):
        self.testoutput=X_test
    def caltestoutput(self):
        self.testoutput=activation(self.activation,np.dot(self.lowerlayer.testoutput,self.weights)-self.b)
    def setalpha(self,alpha):
    	self.alpha = alpha






def derivation(kind,fz):
    size=fz.shape
    if kind == 'Sigmoid':
        return fz*(1-fz)
    if kind == 'tanh':
        return 1-(fz*fz)
    if kind == 'RELU':
        fz[fz>0]=1
        return fz

    print('error')
    exit()
        
def activation(kind,n):
    size=n.shape
    if kind == 'Sigmoid':
        return 1/(1+np.exp(-n))
    if kind == 'RELU':
        return np.maximum(n,np.zeros(size))
    if kind == 'tanh':
        return (np.exp(n)-np.exp(-n))/(np.exp(n)+np.exp(-n))
    print('error')
    exit()