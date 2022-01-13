#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:53:38 2018

@author: sergey
"""
import ephem
import numpy as NP

class BadaryRAO():
    def __init__(self, theDate, observedObject = 'Sun'):
        self.base = 9.8
        self.observatory = ephem.Observer()
        self.observatory.lon = NP.deg2rad(102.217)
        self.observatory.lat = NP.deg2rad(51.759)
        self.observatory.elev= 799
        self.observatory.date= theDate
        if observedObject == 'Moon' or observedObject == 'moon':
            self.obsObject = ephem.Moon()
        else:
            self.obsObject = ephem.Sun()
        self.update()
        
    def update(self):
        self.obsObject.compute(self.observatory)
        noon = self.obsObject.transit_time
        noonText = str(noon).split(' ')[1].split(':')
        declText = str(self.obsObject.dec).split(':')
        self.culmination = float(noonText[0])*3600. + float(noonText[1])*60. + float(noonText[2]) + float(noon) - int(noon)
        self.declination = NP.deg2rad(float(declText[0]) + NP.sign( int(declText[0]))*(float(declText[1])/60. + float(declText[2])/3600.))
        
    def setDate(self, strDate):
        self.observatory.date = strDate
        self.update()

