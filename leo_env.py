# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:38:22 2023

@author: shada
"""

from date_time_array import DateTimeArray
from tle_loader import TLE_Loader
from leo_parameters import LEO_Parameters
from coordinates import GeoCordinates

import numpy as np
import torch

class LeoEnv:
    def  __init__(self, num_satellites = 10, num_features = 3, ue_geo_position=GeoCordinates(-62, 50, 0), del_t=5, num_times=30):
        
        self.action_space = num_satellites
        self.observation_space = num_satellites * num_features
        
        self.ue_geo_position = ue_geo_position
        self.del_t = del_t # simulation time in minutes
        self.num_times = num_times # number of simulation samples
        
        self.steps_before_termination = 0
        
        self.state = np.empty((0, 1))
        
        self.previous_action = torch.zeros(self.action_space)
        
        # define the time window for tracking

        self.date_time_array = DateTimeArray(self.del_t, self.num_times).date_time_array_generate()

        self.leos = TLE_Loader().load_leo_satellites()
        
        self.satellite_names, self.path_loss_matrix, self.elev_matrix, self.serv_time_matrix = LEO_Parameters(self.ue_geo_position, self.date_time_array, self.leos, self.num_times, self.del_t).calculate_leo_parameters()

    def compute_state(self, index):
        
        # index is the current timestamp
        
        # Creating an observation space of 10 x 3
        
        self.state = np.empty((0, int(self.observation_space / self.action_space)))
        
        # Accumlate the path loss for all satellites
        
        path_loss_timestamp = self.path_loss_matrix[:, index]
        
        # print("Path loss for timestamp ",index, path_loss_timestamp)
        
        # Find out the best 10 set of candidate satellites based on coverage
        
        r = np.where(path_loss_timestamp < 200.0)[0]
        
        # print("Eligible satellites ", r)
        
        r_prime  = np.random.choice(np.where(path_loss_timestamp == 200.0)[0], self.action_space - r.shape[0], replace=False)
        
        # print("Complementary satellites ", r_prime)
        
        l = np.concatenate([r,r_prime])
        
        # print("Candidate satellites ", l)
        
        l = np.sort(l)
        
        # print("Sorted candidate satellites ", l)
        
        # List the best 10 candidate satellites 
        
        self.candidate_satellites = np.array(self.satellite_names)[l]
        
        # print("List of candidate satellites ", self.candidate_satellites)
        
        # Gather the path loss for candidate satellites
        
        path_loss = path_loss_timestamp[l]
        
        # print("Path loss for candidate satellites ", path_loss)
        
        # Calculate service time and quality for candidate satellites
        
        avg_elev = np.zeros(self.action_space)
        serv_time = np.zeros(self.action_space)
        
        for i in range(self.action_space):
            # print("Service time ", self.serv_time_matrix[l[i],:], "for candidate satellite ", l[i])
            
            serv_time_first_index = np.nonzero(self.serv_time_matrix[l[i],:])[0][0]
            serv_time_last_index = np.nonzero(self.serv_time_matrix[l[i],:])[0][-1]
            
            # print("Interval of coverage timestamps ", serv_time_first_index, serv_time_last_index)
            
            # print("Index", index, "First Service time index", serv_time_first_index, "Current service time", self.serv_time_matrix[i, index])
            
            if index >= serv_time_first_index and self.serv_time_matrix[l[i], index] > 0.0:
                avg_elev[i] = np.mean (self.elev_matrix[l[i], index:serv_time_last_index+1]) 
                serv_time[i] = np.sum (self.serv_time_matrix[l[i], index:serv_time_last_index+1]) 
            
                # print("Candidate satellite number ", l[i])
                # print("Average elevation angle ", avg_elev[i], "Service time ", serv_time [i])
        
        self.state = np.append(self.state, avg_elev)
        self.state = np.append(self.state, serv_time)
        self.state = np.append(self.state, path_loss)
        
        # print("Computed state ", self.state)
        
        return self.state
        
    def compute_reward (self, action):
        avg_el = self.state[0:self.action_space]
        t_serv = self.state[self.action_space:2*self.action_space]
        path_loss = self.state[2*self.action_space:3*self.action_space]
        
        # print("Average elevation angle ", avg_el, " Service time ", t_serv, " path loss", path_loss)
        # print("Service time ", np.dot(action, t_serv), " path loss", np.dot(action, path_loss), " for action", action)
        if np.dot(t_serv, action) < 1.0 or np.dot(path_loss, action) > 185.0:
            reward = -25.0
        elif torch.all(torch.eq(action, self.previous_action)):
            reward = 25.0
        else:
            reward =  np.dot(action, 10 *(t_serv / self.del_t) + 10 * (avg_el / np.max (avg_el)) - 10 *  (path_loss / np.max(path_loss)))
            
        return reward 
    
    def step(self, action):
        
        # action is a N x 1 vector and state is a N x 3 matrix, 
        # rows should correspond to the satellites
        
        self.state = self.compute_state(self.steps_before_termination)
        
        # print("Computed state ", self.state)
        
        #rewards

        reward = self.compute_reward(action)
        
        # print("Computed reward ", reward)
          
        self.previous_action = action 
        self.steps_before_termination += 1
        
        if (self.steps_before_termination == self.num_times):
            terminated = True
        else:
            terminated = False
            
        return self.state, reward, terminated
    
    def reset(self):
        self.previous_action = torch.zeros(self.action_space)
        self.previous_action[np.random.randint(self.action_space)] = 1.0
        self.state = self.compute_state(0)
        self.steps_before_termination = 1

        return self.state
    
    
    