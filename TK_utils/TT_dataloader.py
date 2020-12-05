import gluonnlp as nlp
from kogpt2.utils import get_tokenizer
import pandas as pd
import logging
import numpy as np

from torch.utils.data import Dataset
U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'

class CharDataset(Dataset):
    def __init__(self, vocab, MAX_LEN=2048):
        self.q_token = U_TKN # BOS os Q
        self.a_token = S_TKN # BOS os A
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.sent_token = SENT
        #-----------------------------------

        self.folder_path = "./TK_data/T0_data"
        self.DATA_PATH = []
        self.DATA_PATH_IDX =[]
        self.DATA_PATH_LEN = []
        self.previous_context = None
        self.MAX_LEN = MAX_LEN

        #self.DATA = pd.read_csv('./TK_data/Chatbot_data/ChatbotData.csv')
        self._tok_path = get_tokenizer()
        self.tokenizer = None
        self.first = True

        self.vocab = vocab
        self.padder = nlp.data.PadSequence(
            MAX_LEN, pad_val=self.vocab[self.vocab.padding_token])

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

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self.DATA_PATH_IDX)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        data_path_idx = self.DATA_PATH_IDX[idx]
        data_path_len = self.DATA_PATH_LEN[idx]
        data_path = self.DATA_PATH[data_path_idx]

        file = open(data_path, 'r', encoding='utf-8')

        if 2 < data_path_len : 
            q_context = self.previous_context
            for i in range(data_path_len-1):
                file.readline()
                
            a = file.readline()
            a_toked = self.tokenizer.encode(a[:-1])
            q_toked = self.bos_token_id + a_toked + self.eos_token_id  
            q_context += q_toked

            self.previous_context = q_context
            q_len = len(q_context)
            if q_len  > self.MAX_LEN:
                raise Exception('None expected q_len : {}'.format(q_len))
                a_len = self.MAX_LEN - q_len
                if a_len <= 0:
                    q_toked = q_toked[-(int(self.MAX_LEN/2)):]
                    q_len = len(q_toked)
                    a_len = self.MAX_LEN - q_len
                    assert a_len > 0
                a_toked = a_toked[:a_len]
                a_len = len(a_toked)
                assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
            pad_token_len = MAX_LEN - q_len
            index_of_words = q_context + self.pad_token_id * pad_token_len
                
        elif data_path_len == 2 :
            q_conext = []
            for i in range(2):
                q = file.readline()
                q_toked = self.tokenizer.encode(q[:-1])
                q_toked = self.bos_token_id + q_toked + self.eos_token_id  
                q_context += q_toked

                self.previous_context = q_context
                q_len = len(q_context)
                if q_len  > self.MAX_LEN:
                    raise Exception('None expected q_len : {}'.format(q_len))
                    a_len = self.MAX_LEN - q_len
                    if a_len <= 0:
                        q_toked = q_toked[-(int(self.MAX_LEN/2)):]
                        q_len = len(q_toked)
                        a_len = self.MAX_LEN - q_len
                        assert a_len > 0
                    a_toked = a_toked[:a_len]
                    a_len = len(a_toked)
                    assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
                pad_token_len = MAX_LEN - q_len
                index_of_words = q_context + self.pad_token_id * pad_token_len

        else :
            raise Exception('None expected lenth of data_path : {}'.format(data_path_len))

        file.close()
        #===========++++ Padding
        if q_len + a_len > self.MAX_LEN:
            a_len = self.MAX_LEN - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.MAX_LEN/2)):]
                q_len = len(q_toked)
                a_len = self.MAX_LEN - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'

        #=============Label
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.maskt,
        ] * q_len + a_toked[1:]

        #==============Mask
        # [0, 0, 0, 0, ....., 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, .... ]
        mask = [0] * q_len + [1] * a_len + [0] * (self.MAX_LEN - q_len - a_len)

        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),
                self.padder(self.vocab[labels]))
