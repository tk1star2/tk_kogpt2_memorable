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
    def __init__(self, vocab, MAX_LEN=32):
        self.DATA = pd.read_csv('./TK_data/Chatbot_data/ChatbotData.csv')
        self._tok_path = get_tokenizer()
        self.tokenizer = None
        self.first = True
        self.q_token = U_TKN # BOS os Q
        self.a_token = S_TKN # BOS os A
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.vocab = vocab
        self.MAX_LEN = MAX_LEN
        self.padder = nlp.data.PadSequence(
            MAX_LEN, pad_val=self.vocab[self.vocab.padding_token])

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        turn = self.DATA.iloc[idx] # read line-------------> 'Q', 'A', 'label' name dtype
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])

        #===========++++ Q token
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)

        #===========++++ A token
        a_toked = [
            self.a_token,
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        a_len = len(a_toked)

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
        '''
        if self.first:
            # 내 월급만 안 올라
            logging.info("contexts : {}".format(q))
            # ['<usr>', '_내', '_월급', '만', '_안', '_올라', '</s>', '<unused1>', '_0', '</s>']
            logging.info("toked ctx: {}".format(q_toked))
            # 다른 분에게도 잘해주는지 살펴봐요.
            logging.info("response : {}".format(a))
            # ['<sys>', '_다른', '_분', '에게도', '_잘', '해주는', '지', '_살펴', '봐', '요', '.', '</s>']
            logging.info("toked response : {}".format(a_toked))
            #['<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','<unused0>','_다른', '_분', '에게도', '_잘', '해주는', '지', '_살펴', '봐', '요', '.', '</s>']
            logging.info('labels {}'.format(labels))
            self.first = False
        '''
        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),
                self.padder(self.vocab[labels]))
