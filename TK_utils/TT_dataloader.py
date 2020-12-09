import gluonnlp as nlp
from kogpt2.utils import get_tokenizer
import pandas as pd
import logging
import numpy as np
import glob

from torch.utils.data import Dataset
U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'

class CharDataset(Dataset):
    def __init__(self, vocab, MAX_LEN=1024):
        self.q_token = U_TKN # BOS os Q
        self.a_token = S_TKN # BOS os A
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.sent_token = SENT
        #-----------------------------------

        self.folder_path = "./TK_data/TT_data"
        self.CONTEXT_IN = []
        self.MASK_IN = []
        self.LABELS_IN = []
        self.MAX_LEN = MAX_LEN

        #self.DATA = pd.read_csv('./TK_data/Chatbot_data/ChatbotData.csv')
        self._tok_path = get_tokenizer()
        self.tokenizer = None
        if self.tokenizer is None:
            self._activate_sp()
        self.first = True

        self.vocab = vocab
        self.padder = nlp.data.PadSequence(
            MAX_LEN, pad_val=self.vocab[self.vocab.padding_token])

        #==========================================================
        for file_path in glob.glob(self.folder_path + "/*.txt"):
            file = open(file_path, 'r', encoding='utf-8')

            even_or_odd = 0;
            CONTEXT_IN = []
            MASK_IN = []
            LABELS_IN = []
            while True:
                data = file.readline()
                print("\n\n\nTK: {}\n\n\n".format(data))
                if not data:
                    break
                q_toked = self.tokenizer(data[:-1])
                #print("S : {}\n".format(data[:-1]))
                if even_or_odd % 2 == 0 :  
                    CONTEXT_IN_TEMP = [self.q_token] + q_toked + [self.eos]
                    CONTEXT_IN += CONTEXT_IN_TEMP
                    MASK_IN += [0] * len(CONTEXT_IN_TEMP)
                    LABELS_IN += [self.maskt] * len(CONTEXT_IN_TEMP)
                else : 
                    CONTEXT_IN_TEMP = [self.a_token] + q_toked + [self.eos]
                    CONTEXT_IN += CONTEXT_IN_TEMP
                    MASK_IN += [1] * len(CONTEXT_IN_TEMP)
                    LABELS_IN += CONTEXT_IN_TEMP
                even_or_odd += 1
            #print("I : {}\n".format(CONTEXT_IN))
            #print("I2 : {}\n".format(MASK_IN))
            CONTEXT_LEN = len(CONTEXT_IN)
            if CONTEXT_LEN  > self.MAX_LEN:
                raise Exception('None expected CONTEXT_LEN : {}'.format(CONTEXT_LEN))
            pad_token_len = MAX_LEN - CONTEXT_LEN
            MASK_IN += [0] * pad_token_len
    
            self.CONTEXT_IN.append(CONTEXT_IN)
            # [0, 0, 0, 0, ....., 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, .... ]
            self.MASK_IN.append(MASK_IN)
            # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
            self.LABELS_IN.append(LABELS_IN)
            file.close()

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self.CONTEXT_IN)

    def __getitem__(self, idx):
        CONTEXT_IN = self.CONTEXT_IN[idx]
        MASK_IN= self.MASK_IN[idx]
        LABELS_IN = self.LABELS_IN[idx]

        '''
        print("======================================================")
        print("0 : {}\n".format(idx))
        print("I : {}\n".format(len(self.CONTEXT_IN)))
        print("I2 : {}\n".format(len(self.MASK_IN)))
        print("I3 : {}\n".format(len(self.LABELS_IN)))
        print("I : {}\n".format(len(CONTEXT_IN)))
        print("I2 : {}\n".format(len(MASK_IN)))
        print("I3 : {}\n".format(len(LABELS_IN)))
        print("======================================================")
        '''
        return (self.padder(self.vocab[CONTEXT_IN]), np.array(MASK_IN),
                self.padder(self.vocab[LABELS_IN]))
'''
            if q_len  > self.MAX_LEN:
                a_len = self.MAX_LEN - q_len
                if a_len <= 0:
                    q_toked = q_toked[-(int(self.MAX_LEN/2)):]
                    q_len = len(q_toked)
                    a_len = self.MAX_LEN - q_len
                    assert a_len > 0
                a_toked = a_toked[:a_len]
                a_len = len(a_toked)
                assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
'''
