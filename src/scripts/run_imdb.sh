# classification, hidden_channels=64
CUDA_LAUNCH_BLOCKING=1 python3 ../main.py \
--n_pretrain 4 \
--seed 42 \
--n_epochs 50 \
--model HAN \
--dataset IMDB \
--debias HTAD \
--dens_scale 0.01 \
--simf cos \
--label_rate 0.05 \
--rel_reg 1000.0 \
--self_reg 3.0 \
--lambda_u 0.05 \
--lambda_t 0.1

# clustering, hidden_channels=128
# CUDA_LAUNCH_BLOCKING=1 python3 ../main.py \
# --n_pretrain 3 \
# --seed 1 \
# --n_epochs 50 \
# --model HAN \
# --dataset IMDB \
# --debias HTAD \
# --dens_scale 0.01 \
# --simf cos \
# --label_rate 0.05 \
# --rel_reg 2000.0 \
# --self_reg 3.0 \
# --lambda_u 0.05 \
# --lambda_t 0.1