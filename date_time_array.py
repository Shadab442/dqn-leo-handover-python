# -*- coding: utf-8 -*-
"""
Created on Sat May  6 20:04:14 2023

@author: shada
"""

from datetime import datetime, timedelta
import numpy as np

class DateTimeArray:
    def __init__(self, del_t = 10, num_times = 30,  start_time = datetime(2023, 4, 29, 9, 30, 0)):
        self.del_t = del_t
        self.num_times = num_times
        self.start_time = start_time
        
    def date_time_array_generate(self):

        date_list = [self.start_time + timedelta(minutes=i) for i in range(self.del_t + 1)]
        date_array = np.array(date_list)

        # Generate linearly spaced datetime vectors

        time_array = np.linspace(date_array[0].timestamp(), date_array[-1].timestamp(), self.num_times)
        return [datetime.fromtimestamp(t) for t in time_array]