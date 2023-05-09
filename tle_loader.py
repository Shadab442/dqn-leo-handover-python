# -*- coding: utf-8 -*-
"""
Created on Sat May  6 20:17:48 2023

@author: shada
"""

import urllib.request
from pyorbital.orbital import Orbital

class TLE_Loader:
    def __init__(self):
        url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
        urllib.request.urlretrieve(url, 'tle.txt')
        
        # read the TLE data and create a list of Orbital objects for all LEO satellites

        with open('tle.txt') as f:
            tle_lines = f.readlines()

        self.tles = [tle_lines[i] for i in range(0, len(tle_lines), 3)]
        
    def load_leo_satellites(self):
        return [Orbital(satellite,tle_file='tle.txt') for satellite in self.tles]