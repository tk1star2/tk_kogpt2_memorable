import torch
import torch.nn as nn
from torch.utils.data import Dataset # 데이터로더

from kogpt2_transformers import get_kogpt2_tokenizer
#from kobert_transformers import get_tokenizer

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0

class WellnessAutoRegressiveDataset(Dataset):
  """Wellness Auto Regressive Dataset"""

  def __init__(self, n_ctx = 1024):
    self.file_path = "./TK_data/T1_wellness/T1_wellness_train.txt"
    self.DATA =[]
    self.tokenizer = get_kogpt2_tokenizer()

    bos_token_id = [self.tokenizer.bos_token_id] # BEGIN of string  <BOS>
    eos_token_id = [self.tokenizer.eos_token_id] # END of string    <EOS>
    pad_token_id = [self.tokenizer.pad_token_id] # OTHER tokens     


    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("    ")

      q = datas[0]
      q_toked = self.tokenizer.encode(q)
      #sentiment = analyser.polarity_scores(text))
      sentiment = vader_polarity(q)
      if sentiment ==1 :
        sentiment = 'g' #good
      else : 
        sentiment = 'b' #bad
      sent_toked = self.tokenizer.encode(sentiment) 
      a = datas[1]
      a_toked = self.tokenizer.encode(a[:-1])

      #===========++++ Q token
      q_toked = bos_token_id + q_toked + eos_token_id + \
                bos_token_id + sent_toked + eos_token_id
      q_len = len(q_toked)

      #===========++++ A token
      #a_toked = bos_token_id + sent_toked + eos_token_id + \
      a_toked = bos_token_id + a_toked + eos_token_id
      a_len = len(a_toked)

      #check padding LEN
      pad_token_len = n_ctx - q_len - a_len

      #===========++++ Padding
      index_of_words = q_toked + a_toked + pad_token_id * pad_token_len

      self.DATA.append(index_of_words)

    file.close()

  def __len__(self):
    return len(self.DATA)

  def __getitem__(self, idx):
    item = self.DATA[idx]
    return item
'''
class WellnessTextClassificationDataset(Dataset):
  """Wellness Text Classification Dataset"""
  def __init__(self,
               file_path = "./data/wellness_dialog_for_text_classification.txt",
               num_label = 359,
               device = 'cpu',
               max_seq_len = 512, # KoBERT max_length
               tokenizer = None
               ):
    self.file_path = file_path
    self.device = device
    self.data =[]
    self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()


    file = open(self.file_path, 'r', encoding='utf-8')

    while True:
      line = file.readline()
      if not line:
        break
      datas = line.split("    ")
      index_of_words = tokenizer.encode(datas[0])
      token_type_ids = [0] * len(index_of_words)
      attention_mask = [1] * len(index_of_words)

      # Padding Length
      padding_length = max_seq_len - len(index_of_words)

      # Zero Padding
      index_of_words += [0] * padding_length
      token_type_ids += [0] * padding_length
      attention_mask += [0] * padding_length

      # Label
      label = int(datas[1][:-1])

      data = {
              'input_ids': torch.tensor(index_of_words).to(self.device),
              'token_type_ids': torch.tensor(token_type_ids).to(self.device),
              'attention_mask': torch.tensor(attention_mask).to(self.device),
              'labels': torch.tensor(label).to(self.device)
             }

      self.data.append(data)

    file.close()

  def __len__(self):
    return len(self.data)
  def __getitem__(self,index):
    item = self.data[index]
    return item
'''
if __name__ == "__main__":
  dataset = WellnessAutoRegressiveDataset()
  #dataset2 = WellnessTextClassificationDataset()
  print(dataset)
  #print(dataset2)
