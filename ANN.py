#!/usr/bin/env python3
"""
describe:artificial_neural_networks
author:strewen
Create On:2018/06/17
"""
import numpy as np
import random
import sys
def sigmoid(x):
    return [1/(1+np.math.exp(-i)) for i in x] 

#将数据分为训练集与测试集
def load_data():
    dataset=np.loadtxt('./fruit.txt',skiprows=1,usecols=[0,3,4,5,6])
    np.random.shuffle(dataset)
    count=len(dataset)
    #划分数据集的位置
    devide=10
    test_x=dataset[0:devide,1:]
    train_x=dataset[devide:,1:]
    test_y=dataset[0:devide,0]
    train_y=dataset[devide:,0]
    return train_x,train_y,test_x,test_y

class ANN:
    def __init__(self,layers,learn_rate,x,y,frequency):
        #各特征值均值列表 形式如[特征1均值，特征2均值...]
        self.mean=None
        #各特征的方差列表，形式如[特征1方差，特征2方差...]
        self.std=None
        #网络层节点数，形式如下：
        #[输入层节点数，隐藏层1节点数，隐藏层2节点数，...,输出层节点数]
        self.layers=layers
        #经过中心化处理的特征集
        self.x=self.normalization(x)
        #经过独热编码的标签集
        self.y=self.oneHot(y)
        #模型学习率
        self.l=learn_rate
        #训练集的迭代次数
        self.frequency=frequency
        #初始化各层间的权重列表，权重都随机初始化在(-0.2~0.2)间
        self.weight=[np.random.uniform(-0.2,0.2,[i,j]) for i,j in zip(layers[1:],layers[:-1])]
        #初始化各网络层见的偏置，偏置都随机初始化在(-0.2~0.2)间
        self.bias=[np.random.uniform(-0.2,0.2,[1,i])for i in layers[1:]]

    #训练数据中心化
    def normalization(self,x):
        data=np.array(x).T
        self.mean=[np.mean(i) for i in data]
        self.std=[np.std(i) for i in data]
        for d in range(len(data)):
            data[d]=(data[d]-self.mean[d])/self.std[d]
        return data.T
    
    #将样本标签进行独热编码处理
    def oneHot(self,y):
        res_y=[]
        for i in y:
            temp=[0]*self.layers[-1];
            temp[int(i-1)]=1
            res_y.append(temp)
        return res_y

    #前向传播
    def feedForword(self,input_data):
        d=input_data
        layer_output=[] #各网络层输出(不包括输入层)
        layer_input=[]  #各网络层输入(不包括输入层)
        for layer in range(len(self.layers)-1):
            layer_input.append(d)
            d=np.dot(d,self.weight[layer].T)+self.bias[layer]
            d=[sigmoid(i) for i in d]
            layer_output.append(d)
        return layer_input, layer_output

    #反向传播
    def backForword(self,input_data,output_data,y):
        layer_weight_count=len(self.weight)    
        for idx,output,in_put in zip(range(layer_weight_count)[::-1],output_data[::-1],input_data[::-1]):
            output=np.array(output)
            in_put=np.array(in_put)
            y=np.array(y)
            #输出层残差计算
            if idx==layer_weight_count-1: 
                err=output*(1-output)*(y-output)
            #隐藏层残差计算
            else :
                err=np.dot(err,self.weight[idx+1])
                err=output*(1-output)*err
            #更新权重与偏置
            for idn in range(len(self.weight[idx])):
                self.weight[idx][idn]=self.weight[idx][idn]+self.l*err[0][idn]*in_put
                self.bias[idx]=(self.bias[idx]+self.l*err)

    #评估函数
    def evaluate(self,x,y):
        correct=0;
        res=self.pred(x)
        for v_y,r_y in zip(y,res):
            if v_y==r_y:
                correct+=1
        return correct/len(y)

    #预测函数
    def pred(self,x):
        res=[]
        #将测试集的特征值标准化
        x=np.array(x).T 
        for i in range(len(x)):
            x[i]=(x[i]-self.mean[i])/self.std[i]
        x=x.T
        for sample in x:
            d=sample
            for layer in range(len(self.layers)-1):
                d=np.dot(d,self.weight[layer].T)+self.bias[layer]
                d=[sigmoid(i) for i in d]
            res.append(np.argmax(d)+1)
        return res
    
    #建立神经网络模型
    def learn(self):
        for i in range(self.frequency):
            for train_x,train_y in zip(self.x,self.y):
                layer_input ,layer_output=self.feedForword(train_x)
                self.backForword(layer_input,layer_output,train_y)
            sys.stdout.write("训练进度：{0}%\r".format(int(i/self.frequency*100)))
            sys.stdout.flush()

def main():
    train_x,train_y,test_x,test_y=load_data()
    clt=ANN([4,100,4],0.1,train_x,train_y,400)
    clt.learn()
    accuracy=clt.evaluate(train_x,train_y)
    print("测试集准确率：%.2f"%clt.evaluate(test_x,test_y)) 
    print("训练集准确率：%.2f"%accuracy)

if __name__=='__main__':
    main()
