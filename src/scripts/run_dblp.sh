# CUDA_LAUNCH_BLOCKING=1 python3 ../main.py \
# --n_pretrain 5 \
# --seed 1 \
# --n_epochs 50 \
# --model HAN \
# --dataset DBLP \
# --debias HTAD \
# --dens_scale 0.01 \
# --simf cos \
# --label_rate 0.05 \
# --rel_reg 2000.0 \
# --self_reg 3.0 \
# --lambda_u 0.05 \
# --lambda_t 0.1

CUDA_LAUNCH_BLOCKING=1 python3 ../main.py \
--n_pretrain 2 \
--seed 1 \
--n_epochs 50 \
--model HAN \
--dataset DBLP \
--debias HTAD \
--dens_scale 0.01 \
--simf cos \
--label_rate 0.05 \
--rel_reg 2000.0 \
--self_reg 3.0 \
--lambda_u 0.02 \
--lambda_t 0.2