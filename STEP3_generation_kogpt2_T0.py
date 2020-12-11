import os
import numpy as np
import torch
from TK_utils.T0_kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

# root_path='drive/My Drive/Colab Notebooks/dialogLM'
save_ckpt_path = f"./TK_checkpoint/kogpt2-T0.pth"
#save_ckpt_path = f"{root_path}/TK_checkpoint/kogpt2-T1_v1.pth"

ctx = "cuda" if torch.cuda.is_available() else "cpu"
print("\n\nTK DEVICE CHECK : {}\n\n".format(ctx))
if ctx=='cpu':
    raise Exception('NOWANT CPU')
device = torch.device(ctx)

save_step = 100 # 학습 저장 주기
learning_rate = 5e-5  # Learning Rate

# STEP2-2. dataset & MODEL
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])
#model.to(device)

#model.eval()
model.train()

# STEP2-3. training configure
tokenizer = get_kogpt2_tokenizer()
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#=========================FOR CONVENIENCE=============================
bos_token_id = [tokenizer.bos_token_id] # BEGIN of string  <BOS>
eos_token_id = [tokenizer.eos_token_id] # END of string    <EOS>
pad_token_id = [tokenizer.pad_token_id] # OTHER tokens     
#====================================================================
# STEP4. evaluation 
while 1:
# for i in range(5):
    sent = input('Question: ')  # '요즘 기분이 우울한 느낌이에요'
    tokenized_indexs = tokenizer.encode(sent)

    q_toked = bos_token_id + tokenized_indexs + eos_token_id
    q_toked_T = torch.tensor(q_toked).unsqueeze(0)

    # set top_k to 50
    sample_output = model.GET_OUTPUT(input_ids=q_toked_T)
    #print("\n\n\nTK0 : {} & {}\n".format(q_toked_T.shape, sample_output.shape)) # T[1,3], T[3,60]
    #print("\n\n\nTK1-1 : {}\n".format(q_toked_T)) # T[[0, 1754, 1]]
    #print("\n\n\nTK1-2 : {}\n".format(sample_output))  # T[[0, 1754, 1, 0, 1335 .... 1]]
    #print("\n\n\nTK2 : {}\n".format(bos_token_id)) #[0]
    #print("\n\n\nTK3 : {}\n".format(eos_token_id)) #[1]
    #print("\n\n\nTK4 : {}\n".format(pad_token_id)) #[3]
    #print("\n\n\nTK5 : {}\n".format(len(tokenized_indexs))) #1
    #print("\n\n\nTK6 : {}\n".format(sample_output[0][len(tokenized_indexs)+2])) #0 ,start
    #sample_output = model(input_ids=q_toked_T)
    #print("\n\n\nTK2 : {}\n".format(sample_output))
    DONE = len(tokenized_indexs)+2
    while 1 :
        if sample_output[0][DONE]==1 :
            break
        else :
            DONE += 1
        
    #print("\n\n\nTK7 : {}\n".format(sample_output[0][DONE])) #1, end
    a_toked_T = sample_output[0][len(tokenized_indexs)+2:DONE+1].unsqueeze(0)
    #print("\n\n\nTK8 : {}\n".format(a_toked_T)) #T[[0, 47674, 8928, 119, 1]]
    #print("\n\n\nTK9 : {}\n".format(a_toked_T.shape)) #T[1, 14]
    qa_toked_T = torch.cat([a_toked_T,q_toked_T],dim=1)

    #print("\n\n\nTK10 : {}\n".format(qa_toked_T.shape)) #1, end
    

    #print("Answer: " + tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:],skip_special_tokens=True))
    print("Answer: " + tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+2:DONE+1],skip_special_tokens=True))

    print(100 * '-')
    #------------------------TK TODO----------------------------
    optimizer.zero_grad()
    #data= qa_toked_T.to(ctx)
    data= qa_toked_T

    #============================================REAL
    prev_outputs = model(data, labels=data)
    _, logits = prev_outputs[:2]
    #============================================END

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = data[..., 1:].contiguous()

    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss.backward()
    optimizer.step()
    #-----------------------------------------------------------

# for s in kss.split_sentences(sent):
#     print(s)
