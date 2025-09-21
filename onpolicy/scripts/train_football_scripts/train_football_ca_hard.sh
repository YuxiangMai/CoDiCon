#!/bin/sh
gpu=$1
seed=4  # 直接设置为特定的 seed 值

# exp param
env="Football"
scenario="academy_counterattack_hard"
algo="cc" # "mappo" "ippo"
exp="rank_seed_${seed}"

# football param
num_agents=4

# train param
num_env_steps=20000000
episode_length=1000

echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=1 python ../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--representation "simple115v2" --rewards "scoring,checkpoints" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 200000 --log_interval 200000 --use_eval --eval_interval 20000 --n_eval_rollout_threads 100 --eval_episodes 32
