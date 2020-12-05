# -*- coding: utf-8 -*-
import argparse
import torch

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from TK_utils.T2_dataloader import CharDataset


parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')
#DATA LOADER
parser.add_argument('--max-len',
                    type=int,
                    default=32,
                    help='max sentence length on input (default: 32)')

parser.add_argument('--batch-size',
                    type=int,
                    default=96,
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
parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')
#???? + max_epochs?


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
#LightingModule : 
#   Required : training_step, tng_dataloader, configure_optimziers 
#   Optional : 
class KoGPT2Chat(LightningModule):
    def __init__(self, args, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = args
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        return output
    #---------------------------------------------------------------------------loop
    def training_step(self, batch, batch_idx):
        #token,     mask, label
        token_ids, mask, label = batch

        out = self(token_ids)

        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, -1e18 * torch.ones_like(out))

        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()

        tensorboard_logs = {'train_loss': loss_avg}

        return {'loss': loss_avg, 'log': tensorboard_logs}

    #---------------------------------------------------------------------------loop
    # 1. pass to model.fit()
    # *2. LightningModule --- train_dataloader
    # 3. LightningDataModule --- train_dataloader

    # This is needed because CharDataset return 3 values
    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        self.train_set = CharDataset(self.vocab, MAX_LEN=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    #------------------------------------------------------------------------optimizer
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        print("\n\n\n\nSO WHAT?!?!?!?!! {}\n\n\n\n".format(self.hparams.max_epochs))

        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    #------------------------------------------------------------------------------------
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor='loss',
        mode='min',
        prefix='kogpt2-T2',
        dirpath='./TK_checkpoint/',
        filename='{epoch:02d}-{loss:.2f}'
    )
    # python train_torch.py --train --gpus 1 --max_epochs 3
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logging.info(args)
    MODEL = KoGPT2Chat(args)
    MODEL.train()

    print("\n\nis this started?1\n\n")
    trainer = Trainer.from_argparse_args(args,
                                        checkpoint_callback=checkpoint_callback, 
                                        gradient_clip_val=1.0)
    print("\n\nis this started?2\n\n")
    #============================================REAL
    trainer.fit(MODEL)
    #============================================END
    logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
