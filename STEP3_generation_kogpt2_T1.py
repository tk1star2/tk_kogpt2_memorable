import os
import numpy as np
import torch
from TK_utils.T1_kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

# root_path='drive/My Drive/Colab Notebooks/dialogLM'
root_path = '.'
data_path = f"{root_path}/TK_data/wellness_dialog_for_autoregressive_train.txt"
save_ckpt_path = f"{root_path}/TK_checkpoint/kogpt2-wellness-auto-regressive.pth"
#save_ckpt_path = f"{root_path}/TK_checkpoint/kogpt2-T1_v1.pth"

ctx = "cuda" if torch.cuda.is_available() else "cpu"
print("\n\nTK DEVICE CHECK : {}\n\n".format(ctx))
if ctx=='cpu':
    raise Exception('NOWANT CPU')
device = torch.device(ctx)

# STEP2-2. dataset & MODEL
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# STEP2-3. training configure
tokenizer = get_kogpt2_tokenizer()


# STEP4. evaluation 
count = 0
output_size = 200 # 출력하고자 하는 토큰 갯수
while 1:
# for i in range(5):
  sent = input('Question: ')  # '요즘 기분이 우울한 느낌이에요'
  tokenized_indexs = tokenizer.encode(sent)

  input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs + [tokenizer.eos_token_id]).unsqueeze(0)
  # set top_k to 50
  sample_output = model.generate(input_ids=input_ids)


  print("Answer: " + \
                tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:len(tokenized_indexs)+5],skip_special_tokens=True), " and ", 
                tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+5:],skip_special_tokens=True)
  )
  print(100 * '-')

# for s in kss.split_sentences(sent):
#     print(s)
