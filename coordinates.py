# -*- coding: utf-8 -*-
"""
Created on Fri May  5 21:54:05 2023

@author: shada
"""

import pyproj

class GeoCordinates:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def geo2ecef (self): 
        # lat, lon in degree and alt in meters, returns in meters
        
        # Define the WGS84 ellipsoid
        geod= pyproj.CRS('EPSG:4326')
        
        # Define the ECEF coordinate system
        ecef = pyproj.CRS('EPSG:4978')
        
        # Define the pyproj transformer for geodetic to ECEF transformation
        transformer = pyproj.Transformer.from_crs(geod, ecef)

        return transformer.transform(self.lat,self.lon,self.alt)