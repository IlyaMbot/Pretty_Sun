#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:45:41 2021

@author: mariagloba
"""
import sys, os

visName = sys.argv[3]
flags = sys.argv[4]
threshold = sys.argv[5]
scan = sys.argv[6]

ANI = {}
for ant in range(32):
    ANI['W%d'%(ant+1)] = str(32 - ant)
for ant in range(64):
    ANI['E%d'%(ant+1)] = str(34 + ant)
for ant in range(31):
    ANI['N%d'%(ant+1)] = str(98 + ant)
ANI['C0'] = '33'

flagdata(vis = visName, antenna = flags)

# imName = 'images/2800'
imName = 'images/' + visName.split('/')[-1].split('.')[0]
tclean(vis = visName,
       imagename = imName,
       niter = 10000,
       threshold = threshold,
       cell = 6,
       imsize = [512,512],
       stokes = 'RRLL',
       usemask = 'pb',
       pbmask = 0.95,
       scan = scan)

imaNameToSave = visName.split('.')[0] + '_clean_image.fit'
exportfits(imagename = imName+'.image', fitsimage = imaNameToSave, overwrite = True)
psfNameToSave = visName.split('.')[0] + '_clean_psf.fit'
exportfits(imagename = imName+'.psf', fitsimage = psfNameToSave, overwrite = True)

clnjunks = ['.flux', '.mask', '.model', '.psf', '.residual','.sumwt','.pb','.image']
for clnjunk in clnjunks:
    if os.path.exists(imName + clnjunk):
        os.system('rm -rf '+imName + clnjunk)