#TRANSFORMER VERSION
#--------------STEP1-----------------
#python3 STEP1_data_generation_T0.py
#python3 STEP1_data_generation_TT.py

#--------------STEP2-----------------
python3 STEP2_train_kogpt2_T0.py
#CUDA_VISIBLE_DEVICES=0 python3 STEP2_train_kogpt2_TT.py --train --gpus 1 --max_epochs 20

#--------------STEP3-----------------
#python3 STEP3_generation_kogpt2_T0.py
#CUDA_VISIBLE_DEVICES=0 python3 STEP3_generation_kogpt2_TT.py
#CUDA_VISIBLE_DEVICES=0 python3 STEP3_generation_kogpt2_TT.py --chat

