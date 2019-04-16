#本项目为蚂蚁金服语义相似的识别的一种基础做法，简单的对采样后的平衡数据集用深度学习的方法判断语义是否相似。

import random
import numpy as np
import torch

from torch import optim
import torch.nn as nn
from sklearn.metrics import f1_score
import data
import model_gru
import torch.nn.functional as F
from Arg import args

torch.manual_seed=(1)
random.seed(1)
#从data源文件中导入项目需要的文件，详情请查看data文件
train_sentence,valid_sentence,test_sentence,word_dict,word2ix,ix2word,word_matrix = data.load_data_and_labels()


args = args()

#定义训练流程，每次训练一个所有训练样本
def trainIters(SiaNetwork,train_sentence,criterion1,batch_size=2000,learning_rate = 0.005):
    SiaNetwork.train()
    SiaNetwork_optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, SiaNetwork.parameters()),
                        lr=learning_rate)

#    SiaNetwork_optimizer = optim.Adam(SiaNetwork.parameters(), lr=learning_rate)

    n_correct=0

    for i in range(0,len(train_sentence),batch_size):
        
        if i+batch_size<=len(train_sentence):
            input_sentences = train_sentence[i:i+batch_size]
        else:
            input_sentences = train_sentence[i:len(train_sentence)]
        
        batch_sentence1,batch_sentence2,batch_labels,sen_lengths1,sen_lengths2= data.get_batch(input_sentences,word2ix)
         
        answer  = SiaNetwork(batch_sentence1,batch_sentence2,sen_lengths1,sen_lengths2)
        
        n = (torch.max(answer.data, 1)[1].cpu().data == batch_labels.cpu().data).numpy()
        n_correct = sum(n)
        
        loss=criterion1(answer, batch_labels)

        SiaNetwork_optimizer.zero_grad()
    
        loss.backward()
    
        SiaNetwork_optimizer.step()

        print("train_acc:",n_correct/1000)
        
        print(i, round(i/len(train_sentence),2),'loss',loss.item())
        
            
#定义预测函数,返回预测类别和真实标签            
def product_result(SiaNetwork,sentence,batch_size):
    SiaNetwork.eval()
    result = []
    labels = []
    

    for i in range(0,len(sentence),batch_size):
        
        if i+batch_size<=len(sentence):
            input_sentences = test_sentence[i:i+batch_size]
        else:
            input_sentences = test_sentence[i:len(sentence)]
        
        batch_sentence1,batch_sentence2,batch_labels,sen_lengths1,sen_lengths2= data.get_batch(input_sentences,word2ix)

        answer  = SiaNetwork(batch_sentence1,batch_sentence2,sen_lengths1,sen_lengths2)
        
        r = list(torch.max(answer.data,1)[1].cpu().numpy())
        t = list(batch_labels.numpy())
        result.extend(r)
        labels.extend(t)
        

    return result,labels

        
#初始化学习率，batch_size等
lr=0.001
mean_l=[]
bs=1000
#初始化的神经网络，详情查看model_gru.py
SiaNetwork = model_gru.SiaNetwork(args,word2ix,word_matrix)
#记录最大准确度，后续训练下降次数
max_acc = 0
d_num = 0

#循环训练神经网络，直到最大训练次数或者验证集连续下降多次

for i in range(10):
    
    trainIters(SiaNetwork=SiaNetwork,train_sentence = train_sentence,criterion1 = nn.CrossEntropyLoss(),batch_size=bs,learning_rate = lr)
    print (i+1,' 15 times had been finished0000000000000000000000000')
    random.shuffle(train_sentence)
    
    result,labels = product_result(SiaNetwork,valid_sentence,bs)
    r = np.array(result)
    t = np.array(labels)
#计算准确度和f1得分    
    acc = sum(t==r)/len(r)
    f1_s =f1_score(t,r,average = "macro")
    print('acc: ',acc,'f1_socres: ',f1_s)
#将验证集结果添加到记录列表中    
    mean_l.append([acc,f1_s])
    

    lr=lr*0.8
#如果某一轮训练效果好，则保存模型
    if acc > max_acc:
       max_acc = acc
       d_num = 0
       torch.save(SiaNetwork.state_dict(),'SiaNetwork')
    else:
        d_num =d_num+1
    if d_num>3:
        break

#加载已保存的模型，并且查看测试集准确度
SiaNetwork.load_state_dict(torch.load('SiaNetwork'))
result,labels = product_result(SiaNetwork,test_sentence,bs)
r = np.array(result)
t = np.array(labels)
acc = sum(t==r)/len(r)
f1_s =f1_score(t,r,average = "macro")

print('acc: ',acc,'f1_socres: ',f1_s)
