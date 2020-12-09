
import argparse
import torch
from gluonnlp.data import SentencepieceTokenizer

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')
parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='TK_checkpoint_temp/kogpt2-TT-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')
#-------------------------------------------------------------------
parser.add_argument('--max-len',
                    type=int,
                    default=1024,#32
                    help='max sentence length on input (default: 32)')

parser.add_argument('--batch-size',
                    type=int,
                    default=1,#96
                    help='batch size for training (default: 96)')
#OPTIMIZER
parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='The initial learning rate')
parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='warmup ratio')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#LightningModule
#   load_from_checkpoint
#       
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from pytorch_lightning.core.lightning import LightningModule
from kogpt2.utils import get_tokenizer

class KoGPT2Chat(LightningModule):
    def __init__(self, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        self._tok_path = get_tokenizer()
        self.previous_context = [[]]
        #self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, NEW=False):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        '''
        self.previous_context += inputs
        if NEW :
            output= self.kogpt2.generate(inputs, do_sample=True, max_length=1024, top_p=0.02,top_k=50, temperature=0.6, no_repeat_ngram_size=None, num_return_sequences=3, early_stopping=False )
            print("{}\n")
            self.previous_context += output
        else :
            output= self.kogpt2.generate(inputs, do_sample=True, max_length=1024, top_p=0.02,top_k=50, temperature=0.6, no_repeat_ngram_size=None, num_return_sequences=3, early_stopping=False )
            print("{}\n")
        ''' 
        return output
    def chat(self, sent='0'):
        tok = SentencepieceTokenizer(self._tok_path, num_best=0, alpha=0)
        sent_tokens = tok(sent)
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
                if q == 'quit':
                    break
                q_tok = tok(q)
                a = ''
                a_tok = []
                while 1:
                    input_ids = torch.LongTensor(
                        [self.vocab[U_TKN]] + self.vocab[q_tok] +
                        self.vocab[EOS, S_TKN] +
                        self.vocab[a_tok]).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = self.vocab.to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                    print("{}\n".format(a))
                    a_tok = tok(a)
                print("Simsimi > {}".format(a.strip()))


if __name__ == "__main__":
    torch.cuda.is_available()
    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n\nTK DEVICE CHECK : {}\n\n".format(ctx))
    if ctx=='cpu':
        raise Exception('NOWANT CPU')
    #ctx = 'cpu'
    device = torch.device(ctx)

    args = parser.parse_args()
    logging.info(args)

    MODEL = KoGPT2Chat.load_from_checkpoint(args.model_params)
    MODEL.chat()
