#!/bin/sh
gpu=$1
map=$2
seed=4 # 直接设置为特定的 seed 值

env="StarCraft2"
algo="cc"

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed}"
exp="rank_var_seed_${seed}"  # 动态修改 exp 变量

CUDA_VISIBLE_DEVICES=${gpu} python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32

