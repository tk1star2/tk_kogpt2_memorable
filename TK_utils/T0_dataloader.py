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
  def __init__(self, MAX_LEN = 1024):
    self.file_path = "./TK_data/T0_data/T0_data.txt"
    self.DATA = []
    self.MAX_LEN = MAX_LEN
    self.signal = 1

    self.tokenizer = get_kogpt2_tokenizer()
    bos_token_id = [self.tokenizer.bos_token_id] # BEGIN of string  <BOS>
    eos_token_id = [self.tokenizer.eos_token_id] # END of string    <EOS>
    pad_token_id = [self.tokenizer.pad_token_id] # OTHER tokens     

    #==========================================================
    file = open(self.file_path, 'r', encoding='utf-8')
    TK_MAX_SIZE = 0
    while True:
      line = file.readline()
      if not line:
        break
      if line == "<CONTEXT_END>\n":
            self.signal = 1
            continue
      datas = line.split("    ")
    

      q_toked = self.tokenizer.encode(datas[0])
      a_toked = self.tokenizer.encode(datas[1][:-1])

      #===========++++ Q token
      q_toked = bos_token_id + q_toked + eos_token_id
      q_len = len(q_toked)

      #===========++++ A token
      a_toked = bos_token_id + a_toked + eos_token_id
      a_len = len(a_toked)

      #check padding LEN
      pad_token_len = MAX_LEN - q_len - a_len
      if pad_token_len < 0:
        continue
      if TK_MAX_SIZE < q_len + a_len:
        TK_MAX_SIZE = q_len+a_len

      #===========++++ Padding
      index_of_words = q_toked + a_toked + pad_token_id * pad_token_len

      self.DATA.append(index_of_words)

    file.close()
    print("\n\n\n MAXSIZE : {}".format(TK_MAX_SIZE))
  def __len__(self):
    return len(self.DATA)

  def __getitem__(self, idx):
    item = self.DATA[idx]
        
    return item


if __name__ == "__main__":
  dataset = WellnessAutoRegressiveDataset()
  #dataset2 = WellnessTextClassificationDataset()
  print(dataset)
  #print(dataset2)
