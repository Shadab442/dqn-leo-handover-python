# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 01:00:17 2023

@author: shada
"""

from leo_dqn_agents import DQN_agent
# from env_cartpole import CartPoleEnv
from leo_env import LeoEnv
import matplotlib.pyplot as plt

class DRL:
    def __init__(self,num_episodes):
        
        # Define the environment
        
        
        # self.env = CartPoleEnv()
        
        self.env = LeoEnv()
        
        state_size = self.env.observation_space
        action_size = self.env.action_space
        
        # Define the DQN agent

        self.agent = DQN_agent(state_size, action_size, n_episodes=num_episodes)
        
        self.steps = 0
        
        self.episodic_mean_loss = []
        self.episodic_mean_reward = []
        
        
    def interact(self, state):
        # Agent chooses an action based on the epsilon greedy policy given the current state
        action = self.agent.choose_action(state) 
        
        # The environment returns reward and next state based on the action taken by the agent
        next_state, reward, done = self.env.step(action)
            
        # Updates the replay buffer, no memory constraints for simplicity
    
        self.agent.update_replay_buffer(state, action, reward, next_state, done)
        
        # Updates the current state
        state = next_state
        
        # Agent learns how to behave in the environment
    
        loss = self.agent.learn()
    
        # Updates the no of steps
        self.steps += 1
        
        # Updates epsilon value
        self.agent.update_epsilon()
        
        return state, reward, done, loss
      
    def episodic_learn(self):
        # Learning the model
        
        for episode in range(self.agent.n_episodes):
        
            # Initialize training parameters
            
            state = self.env.reset()
                
            done = False
            self.steps = 0
            
            iteration_loss = []
            iteration_reward = []
            iteration = 0
            
            while not done:
                state, reward, done, loss = self.interact(state)
                iteration_loss.append(loss)
                
                # print(f"Episode: {episode + 1}/{self.agent.n_episodes}, Iteration: {iteration + 1}, Reward: {reward},Loss: {loss}")
                iteration_reward.append(reward)
                iteration += 1
                
            self.episodic_mean_loss.append(sum(iteration_loss) / len(iteration_loss))
            self.episodic_mean_reward.append (sum(iteration_reward) / len(iteration_reward))
            
            # Updates episodic durations
            self.agent.update_episode_durations(self.steps)
            print(f"Episode: {episode + 1}/{self.agent.n_episodes},  Episodic Mean Reward: {sum(iteration_reward) / len(iteration_reward)}")
            
            # Synchronize target Q network with the policy Q network
            
            self.agent.sync_target_q(episode)
            
    def plot_curves(self):
        # Create a new figure for the CDF of average elevation angles
        plt.figure()
        plt.plot(self.episodic_mean_reward)
        plt.xlabel('Episode index')
        plt.ylabel('Mean reward')
        plt.title('Mean reward vs episode')
        plt.show()