# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 05:53:37 2016

@author: svlesovoi
"""

import numpy as np

def base2uvw(hourAngle, declination, antenna0, antenna1):
    phi = 0.903338787600965
    if ((antenna0 >= 1 and antenna0 <= 128) and (antenna1 >= 129 and antenna1 <= 256)):
        base = np.array([192.5 - antenna1,antenna0 - 64.5,0.])
    elif ((antenna1 >= 1 and antenna1 <= 128) and (antenna0 >= 129 and antenna0 <= 256)):
        base = np.array([192.5 - antenna0,antenna1 - 64.5,0.])
    elif ((antenna0 >= 1 and antenna0 <= 128) and (antenna1 >= 1 and antenna1 <= 128)):
        base = np.array([0.,antenna0 - antenna1,0.])
    elif ((antenna0 >= 129 and antenna0 <= 256) and (antenna1 >= 129 and antenna1 <= 256)):
        base = np.array([antenna0 - antenna1,0.,0.])
    
    base *= 4.9;
    
    phi_operator = np.array([
        [-np.sin(phi), 0., np.cos(phi)],
        [0., 1., 0.],
        [np.cos(phi), 0., np.sin(phi)]
        ])

    uvw_operator = np.array([
        [ np.sin(hourAngle),		 np.cos(hourAngle),		0.	  ],
        [-np.sin(declination)*np.cos(hourAngle),  np.sin(declination)*np.sin(hourAngle), np.cos(declination)], 
        [ np.cos(declination)*np.cos(hourAngle), -np.cos(declination)*np.sin(hourAngle), np.sin(declination)]  
        ])

    return np.dot(uvw_operator, np.dot(phi_operator, base))

_base = 4.9
_phi = 0.903338787600965
_sin_phi = np.sin(_phi)
_cos_phi = np.sin(_phi)
_phi_operator = np.array([
    [-_sin_phi, 0., _cos_phi],
    [0., 1., 0.],
    [_cos_phi, 0., _sin_phi]
    ])

def base2uvwNext(hourAngle, declination, antennaEW):
    sin_H = np.sin(hourAngle)
    cos_H = np.cos(hourAngle)
    sin_D = np.sin(declination)
    cos_D = np.cos(declination)
    
    uvw_operator = np.array([
        [ sin_H,		 cos_H,		0.	  ],
        [-sin_D * cos_H,  sin_D * sin_H, cos_D], 
        [ cos_D * cos_H, -cos_D * sin_H, sin_D]  
        ])

    uvw = []
    for antennaS in np.linspace(192,177,16):
        base = np.array([192.5 - antennaS, antennaEW - 64.5,0.]) * _base
        uvw.append(np.dot(uvw_operator, np.dot(_phi_operator, base)))
    
    return uvw

def S_EW_2_uvw(hourAngle, declination, antennaS):
    phi = 0.903338787600965
    antennasEW = np.linspace(49,80,32)
    antennasS = np.linspace(antennaS,antennaS,32)
    base = np.array([192.5 - antennasS, antennasEW - 64.5,np.zeros(32)]) * 4.9
    
    phi_operator = np.array([
        [-np.sin(phi), 0., np.cos(phi)],
        [0., 1., 0.],
        [np.cos(phi), 0., np.sin(phi)]
        ])

    uvw_operator = np.array([
        [ np.sin(hourAngle),		 np.cos(hourAngle),		0.	  ],
        [-np.sin(declination)*np.cos(hourAngle),  np.sin(declination)*np.sin(hourAngle), np.cos(declination)], 
        [ np.cos(declination)*np.cos(hourAngle), -np.cos(declination)*np.sin(hourAngle), np.sin(declination)]  
        ])
    uvw = np.zeros((3,32))
    for b in range(32):
        uvw[:,b] = np.dot(uvw_operator, np.dot(phi_operator, base[:,b]))
    return uvw
