# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:27:59 2023

@author: shada
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dqn_models import MLP

class DQN_agent:
    def __init__(self, state_space, action_space, lr=1e-3, gamma = 0.6, epsilon_start = 1.0, epsilon_end = 0.05, epsilon_decay = 0.9995, sync_target_net_every = 100, n_episodes = 500, batch_size = 64):
        
        # Q-Learning framework parameters
        
        self.gamma = gamma
        self.lr = lr
        self.epsilon= epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.sync_target_net_every = sync_target_net_every
        
        self.action_space = action_space

        self.n_episodes = n_episodes
        self.batch_size = batch_size
        
        # Define the Deep Q-Network (DQN) model
        
        self.policy_net = MLP(state_space, action_space)
        self.target_net = MLP(state_space, action_space)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss function
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.memory = []
        self.episode_durations = []


    def choose_action(self, state):
        # something was wrong
            
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].view(1, 1)
                action_onehot = torch.zeros(10)
                action_onehot[action] = 1.0
        else:
            action = torch.tensor([[np.random.choice(10)]], dtype=torch.long)
            action_onehot = torch.zeros(10)
            action_onehot[action] = 1.0
    
        return action_onehot
        
        
    def learn(self):
        ret_loss = 0
        if len(self.memory) > self.batch_size:
            # Takes a minibatch of experiences randomly
            
            minibatch = [self.memory[i] for i in np.random.choice(range(len(self.memory)),self.batch_size,replace=False)]

            # Extracts the states, actions, rewards and next states from the pool
            
            states = torch.tensor([x[0] for x in minibatch], dtype=torch.float32)
            actions = torch.tensor([x[1].argmax() for x in minibatch], dtype=torch.long) # something was wrong
            rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float32)
            next_states = torch.tensor([x[3] for x in minibatch], dtype=torch.float32)
            dones = torch.tensor([x[4] for x in minibatch], dtype=torch.bool)
            
            # Calculates the Q-value based on the actions taken

            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) 
            
            # Calculates the target Q-value based on Bellman optimality Equation
            
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

            # Does other training steps like 
            # training loss calculation, 
            # setting gradients to zero
            # Backpropagation and 
            # optimizing the training step
            
            loss = self.criterion(q_values, target_q_values)
            ret_loss = loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()    
            
        return ret_loss
        
    def update_replay_buffer(self,state, action, reward, next_state, done):
        # Updates the replay buffer, no memory constraints for simplicity
        self.memory.append((state, action, reward, next_state, done))
        
    def update_epsilon(self):
        # Updates epsilon value
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        
    def update_episode_durations(self,steps):
        # Updates episodic durations
        self.episode_durations.append(steps)
        
    def sync_target_q(self,episode):        
        # Synchronize target Q network with the policy Q network
        if (episode + 1) % self.sync_target_net_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


       