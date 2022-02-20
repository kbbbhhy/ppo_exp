import os
import datetime

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--critic_discount',type=float,default=0.5,help='discount factor for critic loss in loss function')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=1000, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt,use_cuda=True):
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    model.share_memory()
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available() and use_cuda:
        curr_states = curr_states.cuda()
    curr_episode = 0
    episode_plot = []
    R_plot = []
    acc_plot=[]
    ep_reward_plot = []
    a=set()
    flag_get=False
    start_datetime = datetime.datetime.now().strftime("%m-%d_%H-%M")
    while True:
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            torch.save(model.state_dict(),
                       "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        curr_episode += 1
        episode_plot.append(curr_episode)
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        #flag_get=False
        if flag_get:
            local_steps=int(opt.num_local_steps*0.9)
        else:
            local_steps=opt.num_local_steps
        for _ in range(local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available() and use_cuda:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            if info[0]["flag_get"]==True:
                flag_get=True
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available() and use_cuda:
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
                #done = torch.cuda.FloatTensor(int(done))
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
                #done = torch.FloatTensor(int(done))
            rewards.append(reward)
            dones.append(done)
            curr_states = state
            #if flag_get:
            #    break

        sample_now_tot=len(dones)
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        for i in range(opt.num_epochs):
            indice = torch.randperm(sample_now_tot)
            #indice = torch.randperm(opt.num_local_steps * opt.num_processes)# Returns a random permutation of integers from 0 to n - 1.
            for j in range(opt.batch_size):
                batch_indices = indice[int(j*sample_now_tot/opt.batch_size):int((j+1)*sample_now_tot/opt.batch_size)]
                #batch_indices = indice[
                #                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                #                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[batch_indices]))# Clamps all elements in input into the range [ min, max ]. Letting min_value and max_value be min and max, respectively, this returns:

                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + opt.critic_discount*critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                clip_num=0
                if flag_get:
                    clip_num=20
                else:
                    clip_num=1
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_num)
                optimizer.step()
        print("Episode: {}. Total loss: {}".format(curr_episode, total_loss))
        print(flag_get)
        if flag_get==True:
            a.add(curr_episode)
            print('get')
        if curr_episode>100:
            if curr_episode-100 in a:
                a.remove(curr_episode-100)
            acc=len(a)/100
            print('accuracy is {}'.format(acc))
        else:
            acc=0
        acc_plot.append(acc)
        plt.plot(episode_plot,acc_plot,"r-")
        plt.xlabel('Episode')
        plt.ylabel('Acc 100 recent')
        plt.savefig('Flag_get_acc_{}.pdf'.format(start_datetime))
        plt.close()
        if curr_episode>10000:
            return


if __name__ == "__main__":
    opt = get_args()
    train(opt,use_cuda=True)
