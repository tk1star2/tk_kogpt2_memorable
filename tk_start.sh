#TRANSFORMER VERSION
#--------------STEP1-----------------
#python3 STEP1_data_generation_T0.py
#python3 STEP1_data_generation_T1.py
#python3 STEP1_data_generation_T3.py
#python3 STEP1_data_generation_T5.py

#--------------STEP2-----------------
#python3 STEP2_train_kogpt2_T0.py
#python3 STEP2_train_kogpt2_T1.py
#python3 STEP2_train_kogpt2_T2.py
python3 STEP2_train_kogpt2_TU.py

#--------------STEP3-----------------
#python3 STEP3_generation_kogpt2_T1.py
#python3 STEP3_generation_kogpt2_T2.py

#CUDA_VISIBLE_DEVICES=0 python3 train_torch.py --gpus 1 --train --max_epochs 2

