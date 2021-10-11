#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:00:51 2019

@author: maria
"""

cleanParamsDesc = {'vis = ':'Name of input MS file',
                    'imagename = ':'Name of output image',
                    'cell = ':'pixel weight in arcseconds',
                    'scan = ' : 'scan number or range. Use \'~\' as range operator.',
                    'stokes = ' : '\'I\',\'Q\',\'U\',\'V\',\'IV\',\'QU\',\'IQ\',\'UV\',\'IQUV\',\'RR\',\'LL\',\'XX\',\'YY\',\'RRLL\',\'XXYY\'',
                    'datacolumn = ' : '\'data\' or \'corrected\'',
                    'imsize = ':'size of output image in pixels',
                    'niter = ':'number of iterations',
                    'gain = ' : 'Loop gain',
                    'threshold = ' : 'Stopping threshold (number in units of Jy, or string)',
                    'deconvolver = ' : 'hogbom,clark,multiscale,mtmfs,mem,clarkstokes'}

gaincalParamsDesc = {'vis = ':'Name of input MS file',
                     'caltable = ':'Name of output table',
                     'gaintype = ':'Type of gain solution (G,T,GSPLINE,K,KCROSS)',
                     'calmode = ':'Type of solution: (’ap’-amp&phase, ’p’-phase only, ’a’-amplitude only)',
                     'scan = ':'scan number or range. Use \'~\' as range operator.',
                     'refant = ':'Reference antenna name(s)',
                     'minsnr = ':'Reject solutions below this SNR',
                     'solint = ':'Solution interval'}

applycalParamsDesc = {'vis = ':'Name of input MS file',
                     'caltable = ':'Name of calibration table',
                     'calwt = ':'Use weights?'}