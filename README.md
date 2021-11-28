# Proximal Policy Optimization (PPO) for playing Super Mario Bros

## Introduction

In this project, we successfully train an agent to play super mario bros by using Proximal Policy Optimization (PPO) algorithm **Proximal Policy Optimization Algorithms** [paper](https://arxiv.org/abs/1707.06347).

## Output
You can see the output document to find the test levels and trained_models to find trained models.

### Comparison with human players in warpless mode
The trained models cannot be as good as world record[warpless mode](https://www.speedrun.com/smb1#Warpless). But it achieve good results in some levels.

## How to run

* **Train your model** by running `python train.py`. For example: `python train.py --world 5 --stage 2 --lr 1e-4 --action_type complex`
* **Test your trained model** by running `python test.py`. For example: `python test.py --world 5 --stage 2 --action_type complex --iter 100`

# ppo_exp
