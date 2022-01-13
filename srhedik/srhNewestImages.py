#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 09:41:28 2021

@author: sergeyvlesovoi
"""

from srhFitsFile36 import SrhFitsFile
import srh36MS2
import numpy as NP
import pylab as PL
import os, fnmatch
from astropy.io import fits
from skimage.transform import warp, AffineTransform
from zeep import Client
import scipy.signal
import scipy.constants
import datetime
from ZirinTb import ZirinTb
from optparse import OptionParser

gXSize = 1
gArcsecPerPixel = 1
gIImage = []
gVImage = []
gFrequency = []
gDateObs = []
gImageHist = []

def fitFuncTb(f, A, B, C):
    return A + B*f + C*f**-1.8

def arcmin_format(xy, pos):
  return '%2d' % ((xy - gXSize/2) * gArcsecPerPixel / 60)

def findFits(path, pattern):
    result = []
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename,pattern):
                if os.path.getsize(os.path.join(root,basename)) > 2880:
                    result.append(os.path.join(root,basename))
    return result

def prepareCleanImages(fileName, qSunTb):
    fd = fits.open(fileName)
    
    pAngle = 0
    try:
        client = Client('http://ephemeris.rao.istp.ac.ru/?wsdl')
        result = client.service.Ephemeride('SSRT','sun',fd[0].header['DATE-OBS'])
        pAngle = NP.deg2rad(float(result[0]['PAngle']))
    except:
        pass
    
    rcpImage = fd[0].data[0,0]
    lcpImage = fd[0].data[1,0]

    srh_x_size = fd[0].header['NAXIS1']
    srh_y_size = fd[0].header['NAXIS2']
    
    arcsecPerPixel = fd[0].header['CDELT1']*3600
    resultArcsecPerPixel = 4.911
    resultScale = arcsecPerPixel / resultArcsecPerPixel
    
    global gXSize
    gXSize = srh_x_size
    global gArcsecPerPixel
    gArcsecPerPixel = resultArcsecPerPixel
    
#polarization signum
    iImage = rcpImage + lcpImage
    vImage = rcpImage - lcpImage
    
    iImageHist = NP.histogram(iImage,bins=1000)
    for promFactor in range(5):
        maxInds = scipy.signal.find_peaks(iImageHist[0],prominence = 2000//(promFactor+1))
        if maxInds[0].shape[0] > 1:
            break
    global gImageHist
    gImageHist.append(iImageHist)
    if maxInds[0].shape[0] > 1:
        iImage -= iImageHist[1][maxInds[0][0]]
        iImage /= (iImageHist[1][maxInds[0][1]] - iImageHist[1][maxInds[0][0]])
        vImage /= (iImageHist[1][maxInds[0][1]] - iImageHist[1][maxInds[0][0]])
    else:
        sunLevel = iImage[256-64 : 256+64, 256-64 : 256+64].mean()
        skyLevel = 0
        iImage -= skyLevel
        iImage /= (sunLevel - skyLevel)
        vImage /= (sunLevel - skyLevel)

    iImage *= qSunTb
    vImage *= qSunTb

    scale = AffineTransform(scale=(-resultScale,-resultScale))
    shift = AffineTransform(translation=(-srh_y_size/2,-srh_y_size/2))
    rotate = AffineTransform(rotation = -pAngle)
    back_shift = AffineTransform(translation=(srh_y_size/2,srh_y_size/2))
    
    iImage = warp(iImage,(shift + (rotate + back_shift)).inverse)
    vImage = warp(vImage,(shift + (rotate + back_shift)).inverse)
    iImage = warp(iImage,(shift + (scale + back_shift)).inverse)
    vImage = warp(vImage,(shift + (scale + back_shift)).inverse)
    
    global gIImage
    global gVImage
    
    gIImage.append(iImage)
    gVImage.append(vImage)
    gFrequency.append(fd[0].header['CRVAL3']/1e6 + 0.5)
    
    gDateObs.append(fd[0].header['DATE-OBS'])
    
    pHeader = fits.Header()
    pHeader['DATE-OBS']     = fd[0].header['DATE-OBS']
    pHeader['T-OBS']        = fd[0].header['DATE-OBS']
    pHeader['INSTRUME']     = fd[0].header['INSTRUME']
    pHeader['ORIGIN']       = fd[0].header['ORIGIN']
    pHeader['FREQUENC']     = ('%d') % (fd[0].header['CRVAL3']/1e6 + 0.5)
    pHeader['CDELT1']       = resultArcsecPerPixel
    pHeader['CDELT2']       = resultArcsecPerPixel
    pHeader['CRPIX1']       = srh_x_size // 2
    pHeader['CRPIX2']       = srh_x_size // 2
    pHeader['CTYPE1']       = 'HPLN-TAN'
    pHeader['CTYPE2']       = 'HPLT-TAN'
    pHeader['CUNIT1']       = 'arcsec'
    pHeader['CUNIT2']       = 'arcsec'

    saveFitsIhdu = fits.PrimaryHDU(header=pHeader, data=iImage.astype('float32'))
    saveFitsIpath = 'newest/srh_I_%s_%04d.fit'%(fd[0].header['DATE-OBS'].split('.')[0],fd[0].header['CRVAL3']*1e-6 + .5)
    hduList = fits.HDUList(saveFitsIhdu)
    hduList.writeto(saveFitsIpath, overwrite=True)
    
    saveFitsVhdu = fits.PrimaryHDU(header=pHeader, data=vImage.astype('float32'))
    saveFitsVpath = 'newest/srh_V_%s_%04d.fit'%(fd[0].header['DATE-OBS'].split('.')[0],fd[0].header['CRVAL3']*1e-6 + .5)
    hduList = fits.HDUList(saveFitsVhdu)
    hduList.writeto(saveFitsVpath, overwrite=True)
    
    fd.close()
    
#    T0 = 2e5
#    fig, pl = PL.subplots(figsize=(6,6))
#    fig.tight_layout()
#    pl.clear()
#    pl.xaxis.set_major_locator(PL.MultipleLocator(128))
#    pl.xaxis.set_major_formatter(PL.FuncFormatter(arcmin_format))
#    pl.xaxis.set_minor_locator(PL.MultipleLocator(32))
#    pl.yaxis.set_major_locator(PL.MultipleLocator(128))
#    pl.yaxis.set_major_formatter(PL.FuncFormatter(arcmin_format))
#    pl.yaxis.set_minor_locator(PL.MultipleLocator(32))
#    pl.imshow(iImage,cmap='hot',vmax=T0,vmin=0,origin='lower')
#    levels = NP.linspace(T0,NP.max(iImage),10)
#    pl.contour(iImage,cmap='hot',levels=levels,linewidths=0.5,origin='lower')
#    pl.set_title('SRH I %s,  %d MHz'%(fd[0].header['DATE-OBS'].split('.')[0], fd[0].header['CRVAL3']*1e-6 + .5))
#    pl.set_xlabel('arcmin')
#    pl.set_ylabel('arcmin')
#    fig.savefig(saveFitsIpath.split('.')[0] + '.png')
#
#
#    T0 = 1e5
#    fig, pl = PL.subplots(figsize=(6,6))
#    fig.tight_layout()
#    pl.clear()
#    pl.xaxis.set_major_locator(PL.MultipleLocator(128))
#    pl.xaxis.set_major_formatter(PL.FuncFormatter(arcmin_format))
#    pl.xaxis.set_minor_locator(PL.MultipleLocator(32))
#    pl.yaxis.set_major_locator(PL.MultipleLocator(128))
#    pl.yaxis.set_major_formatter(PL.FuncFormatter(arcmin_format))
#    pl.yaxis.set_minor_locator(PL.MultipleLocator(32))
#    pl.imshow(vImage,cmap='gray',vmax=T0,vmin=-T0,origin='lower')
#    levels = NP.linspace(T0,NP.max(vImage),3)
#    pl.contour(vImage,cmap='gray',levels=levels,linewidths=0.5,origin='lower')
#    levels = NP.linspace(NP.min(vImage),-T0,3)
#    pl.contour(vImage,cmap='gray',levels=levels,linewidths=0.5,origin='lower')
#    pl.set_title('SRH V %s,  %d MHz'%(fd[0].header['DATE-OBS'].split('.')[0], fd[0].header['CRVAL3']*1e-6 + .5))
#    pl.set_xlabel('arcmin')
#    pl.set_ylabel('arcmin')
#    fig.savefig(saveFitsVpath.split('.')[0] + '.png')
#------------------------------------------------------------------------------
parser = OptionParser()
parser.add_option("-f", "--file", dest="fitPath", default = 'newest')
parser.add_option("-t", "--treshold", dest="cleanTresh", default = '0.2mJy')
parser.add_option("-s", "--scan", dest="scan", default = '0~19')
currentDate = datetime.datetime.now().date().strftime("%Y%m%d")

(clean_options, clean_args) = parser.parse_args()
fitPath = clean_options.fitPath
cleanTresh = clean_options.cleanTresh
scan = clean_options.scan

fitNames =  findFits(fitPath,'*.fit')
fitNames.sort()
# fileName = fitNames[-1]

# fileName = fitPath

ZirinQSunTb = ZirinTb()

for fileName in fitNames:
    file = SrhFitsFile(fileName, 1025)
    file.useNonlinearApproach = True
    file.getHourAngle(0)
    frequencyNumber = 3
    for frequency in range(frequencyNumber):
        file.solarPhase(frequency)
        file.updateAntennaPhase(frequency, baselinesNumber = 5)
        file.setFrequencyChannel(frequency)
        file.vis2uv(0, average = 20)
        file.centerDisk()
        print((file.diskLevelLcp, file.diskLevelRcp))
    
        saveName = 'MS/' + fileName.split('/')[-1].split('.')[0] + '_' + str(frequency) + '.ms'
        ms2Table = srh36MS2.SrhMs2(saveName)
        ms2Table.createMS(file, frequencyChannel = [int(frequency)], phaseCorrect = True, amplitudeCorrect = True)
        
        flags_ew_lcp = NP.where(file.ewAntAmpLcp[frequency] == 1e6)[0] + 1
        flags_ew_rcp = NP.where(file.ewAntAmpRcp[frequency] == 1e6)[0] + 1
        flags_ew = NP.unique(NP.append(flags_ew_lcp, flags_ew_rcp))
        flags_n_lcp = NP.where(file.nAntAmpLcp[frequency] == 1e6)[0]+98
        flags_n_rcp = NP.where(file.nAntAmpRcp[frequency] == 1e6)[0]+98
        flags_n = NP.unique(NP.append(flags_n_lcp, flags_n_rcp))
        flags = ','.join(map(str, NP.append(flags_ew, flags_n)))
        
        command = 'casa -c casaclean.py \'' + saveName + '\'  \'' + flags + '\' \'' +  cleanTresh + '\' \'' + scan  + '\''
        os.system(command)
        
        currentDiskTb = ZirinQSunTb.getTbAtFrequncy(file.freqList[frequency]*1e-6)*1e3
        prepareCleanImages(saveName.split('.')[0] + '_clean_image.fit',currentDiskTb)
        
        # clnjunks = ['.ms', '.ms.flagversions', '*clean*.fit']
        # for clnjunk in clnjunks:
        #     if os.path.exists(saveName.split('.')[0] + clnjunk):
        #         o3s.system('rm -rf ' + saveName.split('.')[0] + clnjunk)
        # os.system('rm -rf *casa*.log')
    
    fig, pl = PL.subplots(nrows=frequencyNumber,ncols=2,figsize=(5.6,7.9))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001,wspace=0.001)
    TI = 2e5
    TV = 1e5
    imageDate = gDateObs[0].split('.')[0].split('T')[0].replace('-','')
    imageTime = gDateObs[0].split('.')[0].split('T')[1].replace(':','')[0:4]
    combinedImagePath = 'srh_%s_%s.png'%(imageDate,imageTime)
    for row in range(frequencyNumber):
        pl[row,0].axis('off')
        pl[row,0].imshow(gIImage[row],cmap='hot',vmax=TI,vmin=0,origin='lower')
        levels = NP.linspace(TI,NP.max(gIImage[row]),3)
        pl[row,0].contour(gIImage[row],cmap='hot',levels=levels,linewidths=0.5,origin='lower')
        pl[row,0].text(10,10,'I',color='white')
        if row == 0:
            pl[row,0].text(10,gXSize - 30,'%s'%imageTime,color='white')
    
        pl[row,1].axis('off')
        pl[row,1].imshow(gVImage[row],cmap='gray',vmax=TV,vmin=-TV,origin='lower')
        levels = NP.linspace(TV,NP.max(gVImage[row]),3)
        pl[row,1].contour(gVImage[row],cmap='gray',levels=levels,linewidths=0.5,origin='lower')
        levels = NP.linspace(NP.min(gVImage[row]),-TV,3)
        pl[row,1].contour(gVImage[row],cmap='gray',levels=levels,linewidths=0.5,origin='lower')
        pl[row,1].text(10,10,'V %d MHz'%gFrequency[row],color='white')
    
    fig.savefig(combinedImagePath)
    fdSrhImageName = open('srhImageName.txt','w')
    fdSrhImageName.write(combinedImagePath)
    fdSrhImageName.close()
    gIImage = []
    gVImage = []
    gFrequency = []
    gDateObs = []
    gImageHist = []
