import numpy as np
import jieba 
import torch
from torch.autograd import Variable

import random


#寻找预训练词表中没有的单词，并统文本数据中单词出现的频率
#返回新的单词和低频率词汇的列表
def find_new_word_and_frequency(train_sentence,word_dict,less_frequency):
    
    new_word=[]
    for i in range(0,len(train_sentence)):
        sentence1=train_sentence[i][0][0]
        sentence2=train_sentence[i][0][1]
        new_word.extend([word for word in sentence1 if word not in word_dict])
        new_word.extend([word for word in sentence2 if word not in word_dict])
    
    word_f={}
    for i in new_word:
        if i in word_f:
            word_f[i]+=1
        else:
            word_f[i]=1
    
    word_f_less=[]
    word_f_many=[]
    for key,value in word_f.items():
        if value<less_frequency:
            word_f_less.append(key)
        else:
            word_f_many.append(key)
        
    word_f_less.append(' ')
    word_f_less.append('')

    return new_word ,word_f_less

#从文本中删除与训练词典中没有的单词（这个操作未必都需要做）
def delete_new_word(train_sentence,word_f_less):
    for i in range(len(train_sentence)):
        for j in range(2):
            for word in train_sentence[i][0][j]:
                if word in word_f_less:
                    train_sentence[i][0][j].remove(word)
    return train_sentence


#将所有文本切分开，采样平衡数据集合，得到训练数据，验证数据，测试数据

def get_train_and_valid(train_all_sentence):
    train_same = [];train_dif = []
    for i in range(len(train_all_sentence)):
        if train_all_sentence[i][1]==1:
            train_same.append(train_all_sentence[i])
        else:
            train_dif.append(train_all_sentence[i])
    
    real_train = []          
    real_train_quater=[]
    real_train_quater.extend(train_same)
    real_train_quater.extend(train_dif[:len(train_same)])
    random.shuffle(real_train_quater)
    real_train.extend(real_train_quater)
        
    train_sentence = real_train[:int(0.3*len(real_train))]
    valid_sentence = real_train[int(0.7*len(real_train)):int(0.8*len(real_train))]
    test_sentence = real_train[int(0.8*len(real_train)):]
    return train_sentence ,valid_sentence , test_sentence
        
#利用预训练好的词表， 制造词库矩阵，用来初始化embedding层 
def make_word_matrix(word_dict):
    w_l=[]
    i = 0
    for word in word_dict.keys():
        vec=word_dict[word]
        w_l.append(vec)
        i = i+1
        if i%10000==0:
            print(i,'for total == ',len(word_dict.keys()))
    return torch.tensor(w_l)

#将句子转化为单词的索引列表，并统计长度
    
def Sentence2wordix(sentence,word2ix,max_sentence_len):
    w_l = []
    for word in sentence:
        if word in word2ix.keys():
            w_ix = word2ix[word]
        else:
            w_ix = word2ix['unknow']
        w_l.append(w_ix)
    seq_length = len(w_l)
    while len(w_l)<max_sentence_len:
        w_l.append(word2ix['padding'])
    return w_l,seq_length

#将句子对进行打包，得到一个批次的特征矩阵和标签（训练和预测时，是一次一批一批的将数据输入到网络中的）
def get_batch(input_sentences,word2ix):
    batch_sentence1 = []
    batch_sentence2 = []
    batch_labels = []
    seq_lengths1 = []
    seq_lengths2 = []
    
    max_l1 = 0
    max_l2 = 1
    

    for k in range(len(input_sentences)):
        if len(input_sentences[k][0][0])>max_l1:
            max_l1 = len(input_sentences[k][0][0])
        if len(input_sentences[k][0][1])>max_l2:
            max_l2 = len(input_sentences[k][0][1])            
    for i in range(len(input_sentences)):
        sentence1 = input_sentences[i][0][0]
        sentence2 = input_sentences[i][0][1]
        sentence1_ix, seq_length1= Sentence2wordix(sentence1,word2ix,max_l1)
        sentence2_ix, seq_length2= Sentence2wordix(sentence2,word2ix,max_l2)
        
        batch_sentence1.append(sentence1_ix)
        batch_sentence2.append(sentence2_ix)
        
        seq_lengths1.append(seq_length1)
        seq_lengths2.append(seq_length2)
        
        batch_labels.append(int(input_sentences[i][1]))

    return torch.tensor(batch_sentence1).long(),torch.tensor(batch_sentence2).long(),torch.tensor([batch_labels]).squeeze(0),torch.tensor([seq_lengths1]).squeeze(0),torch.tensor([seq_lengths2]).squeeze(0)

#集成所有的数据处理函数，将原始数据导入，加工得到训练需要的格式
#返回训练集，测试集，验证集，单词词典，单词特征矩阵等    
def load_data_and_labels(train_file_org='atec_nlp_sim_train.csv', train_file_add='atec_nlp_sim_train_add.csv', word_dict_file='word_dict',userdict='userdict1.txt',less_frequency=5):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
#加载结巴分词，如果单词出现次数很少，且不能被字典识别，则不将它作为一个分词单位
    jieba.load_userdict(userdict)
    word_f_less =torch.load('word_f_less')
    for word in word_f_less:
       jieba.del_word(word)
    
    train_data_org = list(open(train_file_org, "r", encoding='utf-8').readlines())
    train_data_add = list(open(train_file_add, "r", encoding='utf-8').readlines())
    train_data=train_data_org+train_data_add
    train_sentence = []
    
    for i in range(0,len(train_data)):
        if train_data[i]:
            seg_list1 = jieba.lcut(train_data[i].split('\t')[1], cut_all=False)
            seg_list2 = jieba.lcut(train_data[i].split('\t')[2], cut_all=False)
            train_sentence.append([[seg_list1,seg_list2],float(train_data[i].split('\t')[-1])])
        else:
            continue
        if i%1000==0:
            print ('transfering sentence to list',i,'/',len(train_data),'had been solved')
    
    
#导入处理后的预训练字典{单词：向量}
    word_dict = torch.load('word_dict')
    word_dict['padding']=np.zeros((200),dtype=np.float64)
    word_dict['unknow']=np.zeros((200),dtype=np.float64)

#构建单词与索引间互相查询的字典    
    word2ix = {word: ix for ix, word in enumerate(word_dict.keys())}
    ix2word = {ix: word for word, ix in word2ix.items()}
    
    
    new_word,word_f_less=find_new_word_and_frequency(train_sentence,word_dict,less_frequency)
    
    train_sentence=delete_new_word(train_sentence,word_f_less)
    word_matrix = make_word_matrix(word_dict).float()
    train_sentence ,valid_sentence,test_sentence = get_train_and_valid(train_sentence)
    
    
    del train_data,train_data_add,train_data_org
    del train_file_add,train_file_org
    
    return train_sentence,valid_sentence,test_sentence,word_dict,word2ix,ix2word,word_matrix






