
import argparse
import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from pytorch_lightning.core.lightning import LightningModule

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')
parser.add_argument('--model_params',
                    type=str,
                    default='TK_checkpoint/kogpt2-T2-last.ckpt',
                    help='model binary for starting chat')
args = parser.parse_args()

#LightningModule
#   load_from_checkpoint
#       
class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        #self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    def chat(self, sent='0'):
        tok = SentencepieceTokenizer(self.train_set._tok_path, num_best=0, alpha=0)
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
                    input_ids = torch.LongTensor([
                        self.vocab[U_TKN]] + self.vocab[q_tok] +
                        self.vocab[EOS, SENT] + self.vocab[sent_tokens] +
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
                    a_tok = tok(a)
                print("Simsimi > {}".format(a.strip()))

if __name__ == "__main__":
    model = KoGPT2Chat.load_from_checkpoint(args.model_params)
    model.chat()
