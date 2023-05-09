# -*- coding: utf-8 -*-
"""
Created on Sat May  6 20:39:38 2023

@author: shada
"""

import numpy as np
import matplotlib.pyplot as plt

from coordinates import GeoCordinates
from NT_link import NT_Link

class LEO_Parameters:
    def __init__(self, ue_geo_position, date_time_array, leos, num_times, del_t):
        self.ue_geo_position = ue_geo_position
        self.date_time_array = date_time_array
        self.leos = leos
        self.num_times = num_times
        self.del_t = del_t
        
        # self.avg_elev  =[]
        # self.serv_time =[]
        self.sat_name =[]
        self.flag = 0
        self.no_act_sat = np.zeros (self.num_times)
        self.timestamps = np.arange(self.num_times) * (20/60)
        
        self.path_loss_matrix = np.empty((0, self.num_times))
        self.elev_matrix = np.empty((0, self.num_times))
        self.serv_time_matrix = np.empty((0, self.num_times))
        
        
    def calculate_leo_parameters(self):
  
        for i,leo in enumerate(self.leos [0:100]):
            
            path_loss_vector = np.ones(self.num_times) * 200.0
            avg_elev_vector = np.zeros(self.num_times)
            serv_time_vector = np.zeros(self.num_times)
            
            for j,time in enumerate(self.date_time_array):
                
                # Location of satellite "leo" at timestamp "time"
                
                sat_lat, sat_lon, sat_alt = leo.get_lonlatalt(time) # altitude in kilometer
                sat_geo_position = GeoCordinates(sat_lat, sat_lon, sat_alt * 10 **3)
                
                # Non-terrestrial link between satellite "leo" and UE
                
                nt_link = NT_Link(sat_geo_position, self.ue_geo_position)
        
                el = nt_link.calculate_elevation_angle ()
                d = nt_link.calculate_distance()
                Lp = nt_link.calculate_path_loss()
                
                # Check if satellite "leo" can provide coverage to UE at timestamp "time"
                
                if el > 10.0:
                    
                    self.no_act_sat [j] += 1
                    path_loss_vector [j] = Lp
                    avg_elev_vector [j] = el
                    serv_time_vector [j] = self.del_t / self.num_times
                    
                    # Check if the satellite starts to provide coverage
                    
                    if self.flag == 0:
                        self.flag = 1
                        
                        # self.avg_elev.append(el)
                        # self.serv_time.append(self.del_t / self.num_times)
                        self.sat_name.append(leo.satellite_name[9:13])
                        
                        
                    # # Check if the satellite continues to provide coverage
                    # else:
                        
                    #     self.avg_elev  [-1] += el
                    #     self.serv_time [-1] += self.del_t  / self.num_times
                    
                    # print(leo.satellite_name, time, sat_geo_position)
                    # print("Path loss", Lp, " Distance",d, " Elevation angle", el)
                        
                
                        
            
            # End of simulation time: Compute average elevation angle for satellite "leo"
            
            if self.flag == 1:
                # self.avg_elev [-1] /= self.serv_time[-1]/(self.del_t  / self.num_times)
                self.flag = 0
                self.path_loss_matrix = np.append(self.path_loss_matrix, [path_loss_vector], axis = 0)
                self.elev_matrix = np.append(self.elev_matrix, [avg_elev_vector], axis = 0)
                self.serv_time_matrix = np.append(self.serv_time_matrix, [serv_time_vector], axis = 0)
            
        # print (self.sat_name, self.path_loss_matrix, self.elev_matrix, self.serv_time_matrix)
        return self.sat_name, self.path_loss_matrix, self.elev_matrix, self.serv_time_matrix
        
        
    def plot_curves(self, avg_elev, serv_time):
        # Sample data
        x = np.arange(len(self.sat_name))
        
        # avg_elev = np.sum (self.elev_matrix, axis = 1) / np.sum (np.nonzero(self.elev_matrix) [1])
        # serv_time = np.sum (self.serv_time_matrix, axis = 1) / np.sum (np.nonzero(self.serv_time_matrix) [1])
    
        # Compute the CDF of average elevation angles
        counts, bin_edges = np.histogram(avg_elev, bins=100, density=True)
        cdf = np.cumsum(counts)
    
        # Create a new figure for the CDF of average elevation angles
        plt.figure()
        plt.plot(bin_edges[1:], cdf/cdf[-1])
        plt.xlabel('Average elevation angle')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of average elevation angle')
        plt.show()
    
        # Compute the CDF of service times
        counts, bin_edges = np.histogram(serv_time, bins=100, density=True)
        cdf = np.cumsum(counts)
    
        # Create a new figure for the CDF of service times
        plt.figure()
        plt.plot(bin_edges[1:], cdf/cdf[-1])
        plt.xlabel('Service time')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of service time')
        plt.show()
    
        # Plot service time and average elevation angle for different satellites
    
        plt.figure()
        plt.stem(x, avg_elev / np.max(avg_elev), linefmt='C0-', markerfmt='C0o', label='Average elevation angle') 
        plt.stem(x, serv_time / np.max(serv_time), linefmt='C1-', markerfmt='C1o', label='Service time')
        plt.xlabel('Satellite index')
        plt.ylabel('Normalized value')
        plt.title('Variation of average elevation angles and service time for different satellites')
        #plt.legend('Average elevation angle','Service time')
        plt.show()
    
    
        # Plot numer of active satellites vs timestamps
    
        plt.figure()
    
        plt.plot(self.timestamps,self.no_act_sat)
        plt.xlabel('Timestamp (minute)')
        plt.ylabel('Number of active satellites')
        plt.title('Numer of active satellites vs timestamps')
    
        plt.show()