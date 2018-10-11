import numpy as np
from temp import layer

def create_layers(alpha,kind,size,n):   #生成神经网络并每层连接
	layers=[]
	for i in range(n+1):
		if i==0:
			layers.append(layer(None,None,[1,1]))
		else:
			layers.append(layer(alpha[i-1],kind[i-1],size[i-1]))

	for i in range(n+1):
		if i == 0:
			layers[0].setlayers(None,layers[1])
		elif i == n:
			layers[n].setlayers(layers[-2],None)
		else:
			layers[i].setlayers(layers[i-1],layers[i+1])

	return layers


def positive_calculation(layers):   #正向计算
	for i in range(np.size(layers)-1):
		layers[i+1].positive()
	return layers

def get_delta(layers,y_train):     #获得每层delta值
	for i in range(np.size(layers)-1):
		if i == 0:
			layers[-1].getdelta(y_train)
		else:
			layers[-i-1].getdelta(0)
	return layers

def update_weights(layers):       #更新每层weights
	for i in range(np.size(layers)-1):
		layers[-i-1].update()

	return layers


def calculate_testoutput(layers): # 计算测试集
	for i in range(np.size(layers)-1):
		layers[i+1].caltestoutput()

	return layers

def reset_alpha(layers,new_alpha):
	for i in range(np.size(layers)-1):
		layers[i+1].setalpha(new_alpha)
	return layers