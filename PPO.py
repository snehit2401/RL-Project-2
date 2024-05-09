import math
import random
import copy
import numpy as np

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, continuous=False, action_std_init=0.6, device=torch.device('cuda')):
        super().__init__()
        
        self.continuous = continuous
        self.device = device

        if self.continuous:
            self.action_dim = action_dim
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(self.device) 

        self.l1 = nn.Linear(state_dim, hidden_size)
        # self.norm1 = nn.BatchNorm1d(hidden_size)
        # self.norm1 = nn.InstanceNorm1d(hidden_size)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        # self.norm2 = nn.BatchNorm1d(hidden_size)
        # self.norm2 = nn.InstanceNorm1d(hidden_size)

        self.l3 = nn.Linear(hidden_size, action_dim)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self, state, softmax_dim = -1):
        # if len(state.shape) == 1:
        #     state = state.unsqueeze(0)
        #     # state = state[None, :]

        x = F.tanh(self.l1(state))
        # x = F.tanh(self.l1(state) if len(state.shape) == 1 else self.norm1(self.l1(state)))
        
        x = F.tanh(self.l2(x))
        # x = F.tanh(self.l2(x) if len(x.shape) == 1 else self.norm2(self.l2(x)))
        
        x = F.tanh(self.l3(x)) if self.continuous else F.softmax(self.l3(x), dim=softmax_dim)
        
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        # self.norm1 = nn.BatchNorm1d(hidden_size)
        # self.norm1 = nn.InstanceNorm1d(hidden_size)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        # self.norm2 = nn.BatchNorm1d(hidden_size)
        # self.norm2 = nn.InstanceNorm1d(hidden_size)

        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        # if len(state.shape) == 1:
        #     state = state.unsqueeze(0)
        #     # state = state[None, :]
        
        x = F.relu(self.l1(state))
        # x = F.relu(self.l1(state) if len(state.shape) == 1 else self.norm1(self.l1(state)))

        x = F.relu(self.l2(x))
        # x = F.relu(self.l2(x) if len(x.shape) == 1 else self.norm2(self.l2(x)))

        x = self.l3(x)

        return x


class PPO:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # A2C Network
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size, self.continuous, self.action_std_init, self.device).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(self.state_dim, self.hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Storing Trajectories
        self.state = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.obs = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)

        self.action = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.action_old_prob = np.zeros((self.T_horizon, 1), dtype=np.float32)

        self.reward = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.end = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        
        if self.continuous:
            self.action_std = self.action_std_init


    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.set_action_std(new_action_std)


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)

        if (self.action_std <= min_action_std):
            self.action_std = min_action_std

        self.set_action_std(self.action_std)


    def select_action(self, state, deterministic=True):
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if self.continuous:
                action_mean = self.actor(state)
                
                if deterministic:
                    return action_mean.detach().cpu().numpy().flatten(), None
                
                cov_mat = torch.diag(self.actor.action_var).unsqueeze(dim=0)
                dist = MultivariateNormal(action_mean, cov_mat)
                
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                
                action = action.detach().cpu().numpy().flatten()
                action_prob = torch.exp(action_logprob).exp().item()
                
            else:
                prob = self.actor(state, softmax_dim=-1)

                if deterministic:
                    action = prob.argmax(dim=-1).item()
                    return action, None

                dist = Categorical(prob)

                action = dist.sample()
                # action_logprob = dist.log_prob(action)
                action_prob = prob.gather(-1, action)

        return action, action_prob


    def evaluate(self, state, action):
        if self.continuous:
            action_mean = self.actor(state)
            
            action_var = self.actor.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        
            action_logprob = dist.log_prob(action)
            action_prob = torch.exp(action_logprob).unsqueeze(dim=-1)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
            
        else:
            prob = self.actor(state, softmax_dim=-1)
            dist = Categorical(prob)
        
            # action_logprob = dist.log_prob(action)
            action_prob = prob.gather(-1, action)
            
        dist_entropy = dist.entropy()
        # dist_entropy = dist.entropy().sum(0, keepdim=True)

        return action_prob, dist_entropy


    def store_transition(self, state, action, reward, obs, action_old_prob, done, end, idx):
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.obs[idx] = obs

        self.action_old_prob[idx] = action_old_prob
        self.done[idx] = done
        self.end[idx] = end
   

    def train(self, curr_steps=0):
        # Exploring decay
        self.entropy_coef *= self.entropy_coef_decay 

        # Preparing Batch 
        state = torch.from_numpy(self.state).to(self.device)
        obs = torch.from_numpy(self.obs).to(self.device)

        action = torch.from_numpy(self.action).to(self.device)
        action_old_prob = torch.from_numpy(self.action_old_prob).to(self.device)

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
        for epoch in range(self.K_epochs):
            # Shuffle the trajectory for better training
            permutation = np.arange(state.shape[0])
            np.random.shuffle(permutation)
            permutation = torch.LongTensor(permutation).to(self.device)

            state, action, td_target, advantage, action_old_prob = \
                state[permutation].clone(), action[permutation].clone(), td_target[permutation].clone(), advantage[permutation].clone(), action_old_prob[permutation].clone()
            
            for i in range(num_batch):
                # Mini-batch
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, state.shape[0]))
                
                action_prob, entropy = self.evaluate(state[index], action[index])

                ratio = torch.exp(torch.log(action_prob) - torch.log(action_old_prob[index]))  # a/b == exp(log(a)-log(b))
                    
                # Clipped surrogate objective
                surr1 = ratio * advantage[index]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage[index]
                
                # Actor Update
                self.actor_optimizer.zero_grad()

                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                actor_loss = actor_loss.mean()

                wandb.log({
                    'Actor/loss': actor_loss,
                    'Actor/curr_steps': curr_steps,
                    'Actor/epoch': epoch,
                })
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40) # Gradient Clipping
                self.actor_optimizer.step()

                # Critic Update
                self.critic_optimizer.zero_grad()

                critic_loss = (self.critic(state[index]) - td_target[index]).pow(2)
                critic_loss = critic_loss.mean()
                
                wandb.log({
                    'Critic/loss': critic_loss,
                    'Critic/curr_steps': curr_steps,
                    'Critic/epoch': epoch,
                })
                
                # # Regularization term
                # for name, param in self.critic.named_parameters():
                #     if 'weight' in name:
                #         critic_loss += param.pow(2).sum() * self.l2_reg

                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40) # Gradient Clipping  
                self.critic_optimizer.step()

    def save(self, env, episode):
        torch.save(self.critic.state_dict(), f'./checkpoints/ppo_critic-{env}-{episode}.pth')
        torch.save(self.actor.state_dict(), f'./checkpoints/ppo_actor-{env}-{episode}.pth')

    def load(self, env, episode):
        self.critic.load_state_dict(torch.load(f'./checkpoints/ppo_critic-{env}-{episode}.pth'))
        self.actor.load_state_dict(torch.load(f'./checkpoints/ppo_actor-{env}-{episode}.pth'))

