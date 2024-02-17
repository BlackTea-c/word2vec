#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# encode分隔符：https://blog.csdn.net/pearl8899/article/details/119328276
#HuggingFace bert  https://huggingface.co/docs/transformers/model_doc/bert
#HuggingFace快速上手（以bert-base-chinese为例） https://zhuanlan.zhihu.com/p/610171544


# In[7]:


import torch
from transformers import BertTokenizer,BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

input_ids = torch.tensor(tokenizer.encode("我是学生")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
sequence_output = outputs[0]
pooled_output = outputs[1]
print(sequence_output.shape)    ## 字向量
print(pooled_output.shape)      ## 句向量

# [cls] 我 是 学 生 [sep]
#  0    1  2  3  4   5


# In[6]:


print(input_ids)  #[CLS]-101   [SEP]-102


# In[5]:


print(sequence_output[0][1]) #“我”的字向量


# In[ ]:




