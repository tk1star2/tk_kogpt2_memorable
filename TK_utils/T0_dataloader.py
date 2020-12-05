import torch
import torch.nn as nn
import glob
import os
from torch.utils.data import Dataset # 데이터로더

from kogpt2_transformers import get_kogpt2_tokenizer
#from kobert_transformers import get_tokenizer

class WellnessAutoRegressiveDataset(Dataset):
  """Wellness Auto Regressive Dataset"""

  #def __init__(self, MAX_LEN = 1024):
  def __init__(self, MAX_LEN = 2048):
    self.folder_path = "./TK_data/T0_data"
    self.DATA_PATH = []
    self.DATA_PATH_IDX =[]
    self.DATA_PATH_LEN = []
    self.previous_context = None
    self.MAX_LEN = MAX_LEN

    self.tokenizer = get_kogpt2_tokenizer()
    self.bos_token_id = [self.tokenizer.bos_token_id] # BEGIN of string  <BOS>
    self.eos_token_id = [self.tokenizer.eos_token_id] # END of string    <EOS>
    self.pad_token_id = [self.tokenizer.pad_token_id] # OTHER tokens     

    TEMP_MAX = 0
    INDEX = 0
    for file_path in glob.glob(self.folder_path + "/*.txt"):
        self.DATA_PATH.append(file_path)
        file = open(file_path, 'r', encoding='utf-8')

        data = file.readline()
        DATA_LEN = 1
        while True:
            data = file.readline()
            DATA_LEN += 1
            #if not line:
            #    break
            self.DATA_PATH_IDX.append(INDEX)
            self.DATA_PATH_LEN.append(DATA_LEN)
        INDEX +=1

  def __len__(self):
    return len(self.DATA)

  def __getitem__(self, idx):
    data_path_idx = self.DATA_PATH_IDX[idx]
    data_path_len = self.DATA_PATH_LEN[idx]
    data_path = self.DATA_PATH[data_path_idx]

    file = open(data_path, 'r', encoding='utf-8')

    if 2 < data_path_len : 
        q_context = self.previous_context
        for i in range(data_path_len-2):
            file.readline()
    elif data_path_len == 2 :
        q_context = []
    else :
        raise Exception('None expected lenth of data_path : {}'.format(data_path_len))

    q = file.readline()
    q_context += self.bos_token_id + self.tokenizer.encode(q[:-1]) + self.eos_token_id
    q_len = len(q_context)
    self.previous_context = q_context


    a = file.readline()
    a_toked = self.bos_token_id + self.tokenizer.encode(a[:-1]) + self.eos_token_id
    a_len = len(a_toked)


    if q_len + a_len > self.MAX_LEN:
        raise Exception('None expected length of q+a : {} > self.MAX_LEN'.format(q_len+a_len, self.MAX_LEN))
        a_len = self.MAX_LEN - q_len
        if a_len <= 0:
            q_context = q_context[-(int(self.MAX_LEN/2)):]
            q_len = len(q_context)
            a_len = self.MAX_LEN - q_len
            assert a_len > 0
        a_toked = a_toked[:a_len]
        a_len = len(a_toked)
        assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'

    pad_token_len = MAX_LEN - q_len - a_len
    index_of_words = q_context + a_toked + self.pad_token_id * pad_token_len
    file.close()
        
    return index_of_words


if __name__ == "__main__":
  dataset = WellnessAutoRegressiveDataset()
  #dataset2 = WellnessTextClassificationDataset()
  print(dataset)
  #print(dataset2)
