

import os
from google.colab import drive
# google mount
drive.mount('/content/gdrive')
os.chdir('./gdrive/My Drive/chatbot/KoGPT2')

! pip3 install transformers

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import torch.nn as nn
import os
import torchtext
import time
import datetime
from tqdm import tqdm
import transformers


#3. DATA 불러오기
# model, tokenizer
model=transformers.GPT2LMHeadModel.from_pretrained('taeminlee/kogpt2')
tokenizer=transformers.GPT2TokenizerFast.from_pretrained('taeminlee/kogpt2')


# special token 추가
special_tokens = {"cls_token":"<cls>"}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
print(tokenizer.all_special_tokens)

class Config(dict):
    __setattr__=dict.__setitem__
    __getattr__=dict.__getitem__

config=Config({'seq_len':64,'bos_idx':tokenizer.eos_token_id,'eos_idx':tokenizer.bos_token_id,'pad_idx':tokenizer.pad_token_id,'cls_idx':tokenizer.encode('<cls>')[0],'n_vocab':tokenizer.vocab_size+1, 'batch_size':2})

# bos token, eos token 붙이기
class make_data_set:
    def __init__(self,config):
        '''
        data : data frame
        data - Q, A 로 구성 
        '''
        data=pd.read_csv('https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData%20.csv',sep=',',header=0)
        data['Q']='<s>'+data['Q']+'<cls>' #  start token, cls token add 
        data['A']='<s>'+data['A'] # start token, end token add
        
        data['Q_input_id']=data['Q'].apply(lambda i : tokenizer.encode(i))
        data['A_input_id']=data['A'].apply(lambda i : tokenizer.encode(i))
        
        data['attention_mask']=None
        data['token_id']=None
        data['cls_position']=None
        Max = config.seq_len
        for i in data.index:
            if len(data.Q_input_id[i])+len(data.A_input_id[i])>Max:
                data.Q_input_id[i]=data.Q_input_id[i][-(Max//2):]
                data.A_input_id[i]=data.A_input_id[i][:(Max-(Max//2))]
            data['attention_mask'][i] = [1] * (len(data.Q_input_id[i])+len(data.A_input_id[i]))+[0] * (Max - (len(data.Q_input_id[i])+len(data.A_input_id[i])))
            data['token_id'][i] = [0] * len(data.Q_input_id[i]) + [1] * (Max-len(data.Q_input_id[i]))
            data['cls_position'][i] = len(data.Q_input_id[i])-1
        data['input_id']=data['Q_input_id']+data['A_input_id']
        data['input_id']=data['input_id'].apply(lambda j : j+(Max-len(j))*[tokenizer.pad_token_id])
        data['lm_label']=data['Q_input_id'].apply(lambda i : len(i) * [tokenizer.pad_token_id])+data['A_input_id'].apply(lambda i : i[1:]+[tokenizer.eos_token_id])
        data['lm_label']=data['lm_label'].apply(lambda j : j+(Max-len(j))*[tokenizer.pad_token_id])
        self.data=data

    def return_data_set(self):
        result=self.data.loc[:,['input_id','attention_mask','token_id','cls_position','lm_label','label']]
        return result
# gpt2 관련 input들 처리하기
# index of words - tokenized되고 encode 된 id들
# token type ids - [0] : question [1] : answer 
# attention mask - [1] : padding 안 된 부분 [0] : padding 된 부분
# label : Q 
# cls position : cls의 위치 (index)

make_data=make_data_set(config)
dataset=make_data.return_data_set()

dataset.to_csv('./data.csv')

from sklearn.model_selection import train_test_split
train,test=train_test_split(dataset,test_size=0.2)

from torch.utils.data import DataLoader, TensorDataset
train_dataset=TensorDataset(torch.LongTensor(train.input_id.values.tolist()), \
                            torch.LongTensor(train.attention_mask.values.tolist()), \
                            torch.LongTensor(train.token_id.values.tolist()), \
                            torch.LongTensor(train.cls_position.values.tolist()), \
                            torch.LongTensor(train.lm_label.values.tolist()),\
                            torch.LongTensor(train.label.values.tolist()))
test_dataset=TensorDataset( torch.LongTensor(test.input_id.values.tolist()),\
                            torch.LongTensor(test.attention_mask.values.tolist()), \
                            torch.LongTensor(test.token_id.values.tolist()), \
                            torch.LongTensor(test.cls_position.values.tolist()), \
                            torch.LongTensor(test.lm_label.values.tolist()), \
                            torch.LongTensor(test.label.values.tolist()))

# train loader, test loader
train_loader=DataLoader(train_dataset,batch_size=2,drop_last=True)
test_dataset=DataLoader(test_dataset,batch_size=2,drop_last=True)

class my_kogpt2_model(nn.Module):
    def __init__(self, config, kogpt2):
        super().__init__()
        self.config = config
        self.model = kogpt2
        self.classification = nn.Linear(self.config.n_vocab, 3)
    
    def forward(self,data):
        input_ids=data[0]
        attention_mask=data[1]
        token_type_ids=data[2]
        cls_position=data[3] # batch size 
        lm_label=data[4]
        label=data[-1]
        output = self.model.forward(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=True)
        lm_output = output.logits # batch size, seq len, n_vocab -> batch size, n_vocab
        classification_output = torch.vstack([self.classification.forward(lm_output[_,cls_position[_]]) for _ in range(self.config.batch_size)])
        return lm_output, classification_output

#학
import time,datetime
def Eplased(dt):
    d=int(round(dt))
    return str(datetime.timedelta(seconds=d))


device='cuda' if torch.cuda.is_available() else 'cpu'
epochs = 50         # Num of Epoch
learning_rate = 1e-5
Model = my_kogpt2_model(config, model)
Model.to(device)
optimizer = transformers.AdamW(model.parameters(),lr=learning_rate) 
criterion1 = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,reduction='sum') # language modeling
criterion2 = nn.CrossEntropyLoss() # classification


model.train()
start_time = time.time()
for epoch in tqdm(range(1,epochs+1),desc='epoch',mininterval=3600):
    total_loss_1=0.
    total_loss_2=0.
    for step, data in tqdm(enumerate(train_loader),desc='step',mininterval=600):
        optimizer.zero_grad()
        data = tuple(i.to(device) for i in data)
        a,b=Model.forward(data)        
        loss1=criterion1(a.reshape(-1,50001),data[4].reshape(-1))
        loss2=criterion2(b,data[-1])
        loss=loss1/config.batch_size+loss2/config.batch.size
        loss.backward()
        total_loss_1+=(loss1/config.batch_size).item()
        total_loss_2+=(loss2).item()
        optimizer.step()
    print('epoch : %d'%epoch)
    print('total loss 1 : %.3f'%(total_loss_1/len(train_loader)))
    print('total loss 2 : %.3f'%(total_loss_2/len(train_loader)))
    print('Eplased Time : %s'%(Eplased(time.time()-start_time)))


torch.save(model.state_dict(),'./chatbot_transformer')


