
'''
from google.colab import drive
drive.mount('/content/drive')

!ls drive/'My Drive'/'Colab Notebooks'/

!pip install -r drive/'My Drive'/'Colab Notebooks'/dialogLM/requirements.txt

import sys
sys.path.append('drive/My Drive/Colab Notebooks/')
'''



import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import dataloader
from TK_utils.T1_dataloader import WellnessAutoRegressiveDataset
from TK_utils.T1_kogpt2 import DialogKoGPT2

torch.cuda.is_available()

root_path='.'
data_path = f"{root_path}/TK_data/wellness_dialog_dataset/wellness_dialog_for_autoregressive_train.txt"
save_ckpt_path = f"{root_path}/TK_checkpoint/kogpt2-T1.pth"



# STEP2-1. training configure
n_epoch = 5         # Num of Epoch
batch_size = 2      # 배치 사이즈
ctx = "cuda" if torch.cuda.is_available() else "cpu"

print("\n\nTK DEVICE CHECK : {}\n\n".format(ctx))
if ctx=='cpu':
    raise Exception('NOWANT CPU')
device = torch.device(ctx)

save_step = 100 # 학습 저장 주기
learning_rate = 5e-5  # Learning Rate


# STEP2-2. dataset & MODEL
#dataset= WellnessAutoRegressiveDataset(data_path)
dataset= WellnessAutoRegressiveDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DialogKoGPT2()
model.to(device)


# STEP2-3. training configure
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# STEP3. training 
losses =[]
for epoch in range(n_epoch):
    count = 0
    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = torch.stack(data)  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
            data = data.transpose(1, 0)
            data= data.to(ctx)

            #============================================REAL
            outputs = model(data, labels=data)
            _, logits = outputs[:2]
            #============================================END

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = data[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # if count % 10 == 0:
            #     print('epoch no.{} train no.{}  loss = {}'.format(epoch, count + 1, loss))
            if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
                torch.save({
                    'epoch': epoch,
                    'train_no': count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, save_ckpt_path)
            count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")




