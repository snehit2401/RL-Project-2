import math
import random
import copy
import numpy as np

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state, softmax_dim = 0):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))

        prob = F.softmax(self.l3(x), dim=softmax_dim)
        return prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()

        self.C1 = nn.Linear(state_dim, hidden_size)
        self.C2 = nn.Linear(hidden_size, hidden_size)
        self.C3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))

        v = self.C3(v)
        return v


class PPO:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # A2C Network
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.state_dim, self.hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Storing Trajectories
        self.state = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.obs = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)

        self.action = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.action_logprob = np.zeros((self.T_horizon, 1), dtype=np.float32)

        self.reward = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.end = np.zeros((self.T_horizon, 1), dtype=np.bool_)


    def select_action(self, state, deterministic=True):
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            pi = self.actor(state, softmax_dim=0)

            if deterministic:
                action = torch.argmax(pi).item()
                return action, None
            else:
                m = Categorical(pi)
                action = m.sample().item()

                pi_action = pi[action].item()
                return action, pi_action


    def store_transition(self, state, action, reward, obs, action_logprob, done, end, idx):
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.obs[idx] = obs

        self.action_logprob[idx] = action_logprob
        self.done[idx] = done
        self.end[idx] = end
   

    def train(self):
        # Exploring decay
        self.entropy_coef *= self.entropy_coef_decay 

        # Preparing Batch 
        state = torch.from_numpy(self.state).to(self.device)
        obs = torch.from_numpy(self.obs).to(self.device)

        action = torch.from_numpy(self.action).to(self.device)
        action_logprob = torch.from_numpy(self.action_logprob).to(self.device)

        reward = torch.from_numpy(self.reward).to(self.device)
        done = torch.from_numpy(self.done).to(self.device)
        end = torch.from_numpy(self.end).to(self.device)

        ## Compute Advantage and TD target
        with torch.no_grad():
            value_state = self.critic(state)
            value_obs = self.critic(obs)

            deltas = reward + self.gamma * value_obs * (~done) - value_state
            deltas = deltas.cpu().flatten().numpy()
            arr_advantage = [0]

            for delta, end in zip(deltas[::-1], end.cpu().flatten().numpy()[::-1]):
                advantage = delta + self.gamma * self.lambd * arr_advantage[-1] * (~end)
                arr_advantage.append(advantage)

            arr_advantage.reverse()

            arr_advantage = copy.deepcopy(arr_advantage[0:-1])
            advantage = torch.tensor(arr_advantage).unsqueeze(1).float().to(self.device)

            td_target = advantage + value_state
            # if self.advantage_normalization:
            #     advantage = (advantage - advantage.mean()) / ((advantage.std() + 1e-4))  # sometimes helps


        ## PPO Update
        num_batch = int(math.ceil(state.shape[0] / self.batch_size)) 

        # Slice long trajectopy into short trajectory and perform mini-batch PPO update
        for _ in range(self.K_epochs):
            # Shuffle the trajectory for better training
            permutation = np.arange(state.shape[0])
            np.random.shuffle(permutation)
            permutation = torch.LongTensor(permutation).to(self.device)

            state, action, td_target, advantage, action_logprob = \
                state[permutation].clone(), action[permutation].clone(), td_target[permutation].clone(), advantage[permutation].clone(), action_logprob[permutation].clone()

            for i in range(num_batch):
                # Mini-batch
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, state.shape[0]))

                prob = self.actor(state[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_action = prob.gather(1, action[index])
                ratio = torch.exp(torch.log(prob_action) - torch.log(action_logprob[index]))  # a/b == exp(log(a)-log(b))
                
                # Clipped surrogate objective
                surr1 = ratio * advantage[index]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage[index]
                
                # Actor Update
                self.actor_optimizer.zero_grad()

                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                actor_loss = actor_loss.mean()

                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                # Critic Update
                self.critic_optimizer.zero_grad()

                critic_loss = (self.critic(state[index]) - td_target[index]).pow(2)
                critic_loss = critic_loss.mean()
                
                # Regularization term
                # for name, param in self.critic.named_parameters():
                #     if 'weight' in name:
                #         critic_loss += param.pow(2).sum() * self.l2_reg

                critic_loss.backward()
                self.critic_optimizer.step()

    def save(self, env, episode):
        torch.save(self.critic.state_dict(), f'./checkpoints-old/ppo_critic-{env}-{episode}.pth')
        torch.save(self.actor.state_dict(), f'./checkpoints-old/ppo_actor-{env}-{episode}.pth')

    def load(self, env, episode):
        self.critic.load_state_dict(torch.load(f'./checkpoints-old/ppo_critic-{env}-{episode}.pth'))
        self.actor.load_state_dict(torch.load(f'./checkpoints-old/ppo_actor-{env}-{episode}.pth'))

