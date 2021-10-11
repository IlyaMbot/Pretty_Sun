#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 04:27:13 2018

@author: sergey
"""

import casacore.tables as T
import numpy as NP
from astropy.time import Time, TimeDelta
import base2uvw as UVW

class SrhMs2():
    def __init__(self, tableName):
        a = 6378137.0000;
        b = 6356752.3141;
        e2 = (a*a - b*b) / (a*a);
        ssrtLat = NP.deg2rad(51.  + 45.562/60.);
        ssrtLon = NP.deg2rad(102. + 13.160/60.);
        ssrtHeight = 832.;
        v = a / (NP.sqrt(1. - e2*(NP.sin(ssrtLat) * NP.sin(ssrtLat))));
        self.x = (v + ssrtHeight)*NP.cos(ssrtLat)*NP.cos(ssrtLon);
        self.y = (v + ssrtHeight)*NP.cos(ssrtLat)*NP.sin(ssrtLon);
        self.z = (((1. - e2)*v) + ssrtHeight)*NP.sin(ssrtLat);
        
        self.dataTable = T.default_ms(name = tableName)
        self. antennaTable = T.default_ms_subtable('ANTENNA',name = tableName + '/ANTENNA')
        self.spectralWindowTable = T.default_ms_subtable('SPECTRAL_WINDOW',name = tableName + '/SPECTRAL_WINDOW')
        self.dataDescriptionTable = T.default_ms_subtable('DATA_DESCRIPTION',name = tableName + '/DATA_DESCRIPTION')
        self.polarizationTable = T.default_ms_subtable('POLARIZATION',name = tableName + '/POLARIZATION')
        self.sourceTable = T.default_ms_subtable('SOURCE',name = tableName + '/SOURCE')
        self.fieldTable = T.default_ms_subtable('FIELD',name = tableName + '/FIELD')
        self.feedTable = T.default_ms_subtable('FEED',name = tableName + '/FEED')
        self.observationTable = T.default_ms_subtable('OBSERVATION',name = tableName + '/OBSERVATION')

    def initDataTable(self, srhFits, frequencyChannel, ewLcpPhaseCorrection, ewRcpPhaseCorrection, sLcpPhaseCorrection, sRcpPhaseCorrection, phaseCorrect = True, amplitudeCorrect = False):
        declination = srhFits.getDeclination()
        noon = srhFits.RAO.culmination
        
        visibilityNumber = 512
        
        dataDesc = T.makearrcoldesc('DATA',0.+0j, shape=[1,2],valuetype='complex')
        correctedDataDesc = T.makearrcoldesc('CORRECTED_DATA',0.+0j, shape=[1,2],valuetype='complex')
        self.dataTable.addcols(dataDesc)
        self.dataTable.addcols(correctedDataDesc)
        uv = NP.zeros((1,2),dtype='complex64')
        flag = NP.zeros((1,2),dtype='bool')
        weight = NP.array([100.,100.],dtype='float32')
        sigma = NP.array([.01,.01],dtype='float32')
       
        fitsDate =srhFits.hduList[0].header['DATE-OBS'].replace('/','-') + 'T00:00:00';

        for scan in range(srhFits.dataLength):
            scanDate = Time(fitsDate, format='isot',scale='utc');
            scanTime = srhFits.freqTime[frequencyChannel, scan]
            scanDate += TimeDelta(scanTime,format='sec')
            hourAngle = NP.deg2rad((scanTime - noon)*15./3600.)
            for vis in range(visibilityNumber):
                self.dataTable.addrows()
                row = scan*visibilityNumber + vis;
                i = vis // 32
                j = vis % 32 
                uv[0,0] = srhFits.visLcp[frequencyChannel,scan,vis].copy()
                uv[0,1] = srhFits.visRcp[frequencyChannel,scan,vis].copy()
                if phaseCorrect:
                    uv[0,0] *= NP.exp(1j*(srhFits.ewAntPhaLcp[j] - srhFits.sAntPhaLcp[i] + ewLcpPhaseCorrection[j] + sLcpPhaseCorrection[i]))
                    uv[0,1] *= NP.exp(1j*(srhFits.ewAntPhaRcp[j] - srhFits.sAntPhaRcp[i] + ewRcpPhaseCorrection[j] + sRcpPhaseCorrection[i]))
                if amplitudeCorrect:
                    uv[0,0] *= 0.01*NP.sqrt(srhFits.ampLcp[frequencyChannel,scan,j + 16] * srhFits.ampLcp[frequencyChannel,scan,i])
                    uv[0,1] *= 0.01*NP.sqrt(srhFits.ampRcp[frequencyChannel,scan,j + 16] * srhFits.ampRcp[frequencyChannel,scan,i])
                    uv[0,0] /= (srhFits.ewAntAmpLcp[j] * srhFits.sAntAmpLcp[i])
                    uv[0,1] /= (srhFits.ewAntAmpRcp[j] * srhFits.sAntAmpRcp[i])

                self.dataTable.col('SCAN_NUMBER')[row] = scan
                self.dataTable.col('ANTENNA1')[row] = srhFits.antennaA[vis] - 49
                self.dataTable.col('ANTENNA2')[row] = 192 - srhFits.antennaB[vis] + 32
                self.dataTable.col('ARRAY_ID')[row] = 0
                self.dataTable.col('STATE_ID')[row] = -1
                self.dataTable.col('PROCESSOR_ID')[row] = -1
                self.dataTable.col('FEED1')[row] = 0
                self.dataTable.col('FEED2')[row] = 0
                self.dataTable.col('WEIGHT')[row] = weight
                self.dataTable.col('SIGMA')[row] = sigma
                self.dataTable.col('EXPOSURE')[row] = 0.28
                self.dataTable.col('INTERVAL')[row] = 0.28
                self.dataTable.col('FLAG')[row] = flag
                self.dataTable.col('FLAG_ROW')[row] = False
                self.dataTable.col('DATA_DESC_ID')[row] = 0
                self.dataTable.col('TIME')[row] = scanDate.mjd*(24.*3600)
                self.dataTable.col('TIME_CENTROID')[row] = scanDate.mjd*(24.*3600)
                
                self.dataTable.col('DATA')[row] = uv
                self.dataTable.col('CORRECTED_DATA')[row] = uv
                self.dataTable.col('UVW')[row] = -1. * UVW.base2uvw(hourAngle,declination,srhFits.antennaA[vis], srhFits.antennaB[vis])#, correct_positions = False)
        self.dataTable.close()
        
    def initDataTableRedundant(self, srhFits, frequencyChannel, doubleBaselines = False):
        declination = srhFits.getDeclination()
        noon = srhFits.RAO.culmination
        
        dataDesc = T.makearrcoldesc('DATA',0.+0j, shape=[1,2],valuetype='complex')
        correctedDataDesc = T.makearrcoldesc('CORRECTED_DATA',0.+0j, shape=[1,2],valuetype='complex')
        self.dataTable.addcols(dataDesc)
        self.dataTable.addcols(correctedDataDesc)
        uv = NP.zeros((1,2),dtype='complex64')
        flag = NP.zeros((1,2),dtype='bool')
        weight = NP.array([100.,100.],dtype='float32')
        sigma = NP.array([.01,.01],dtype='float32')
       
        fitsDate =srhFits.hduList[0].header['DATE-OBS'].replace('/','-') + 'T00:00:00';

        if doubleBaselines:
            visibilityNumber = 91
        else:
            visibilityNumber = 47
            
        for scan in range(srhFits.dataLength):
            scanDate = Time(fitsDate, format='isot',scale='utc');
            scanTime = srhFits.freqTime[frequencyChannel, scan]
            scanDate += TimeDelta(scanTime,format='sec')
            hourAngle = NP.deg2rad((scanTime - noon)*15./3600.)
            for n in range(visibilityNumber):
                self.dataTable.addrows()
                row = scan*visibilityNumber + n;
                vis = 512 + n
                if n != visibilityNumber-1: 
                    row = scan*visibilityNumber + n;
                    uv[0,0] = srhFits.visLcp[frequencyChannel,scan,vis] 
                    uv[0,1] = srhFits.visRcp[frequencyChannel,scan,vis]
                    self.dataTable.col('ANTENNA1')[row] = self.antennaName2Index(srhFits.antennaA[vis])
                    self.dataTable.col('ANTENNA2')[row] = self.antennaName2Index(srhFits.antennaB[vis])
                else:
                    row = scan*visibilityNumber + n;
                    vis = 15
                    uv[0,0] = srhFits.visLcp[frequencyChannel,scan,vis]
                    uv[0,1] = srhFits.visRcp[frequencyChannel,scan,vis]
                    self.dataTable.col('ANTENNA1')[row] = self.antennaName2Index(srhFits.antennaA[vis])
                    self.dataTable.col('ANTENNA2')[row] = self.antennaName2Index(srhFits.antennaB[vis])
                self.dataTable.col('SCAN_NUMBER')[row] = scan
                self.dataTable.col('ARRAY_ID')[row] = 0
                self.dataTable.col('STATE_ID')[row] = -1
                self.dataTable.col('PROCESSOR_ID')[row] = -1
                self.dataTable.col('FEED1')[row] = 0
                self.dataTable.col('FEED2')[row] = 0
                self.dataTable.col('WEIGHT')[row] = weight
                self.dataTable.col('SIGMA')[row] = sigma
                self.dataTable.col('EXPOSURE')[row] = 0.28
                self.dataTable.col('INTERVAL')[row] = 0.28
                self.dataTable.col('FLAG')[row] = flag
                self.dataTable.col('FLAG_ROW')[row] = False
                self.dataTable.col('DATA_DESC_ID')[row] = 0
                self.dataTable.col('TIME')[row] = scanDate.mjd*(24.*3600)
                self.dataTable.col('TIME_CENTROID')[row] = scanDate.mjd*(24.*3600)
                
                self.dataTable.col('DATA')[row] = uv
                self.dataTable.col('CORRECTED_DATA')[row] = uv
                self.dataTable.col('UVW')[row] = -1. * UVW.base2uvw(hourAngle,declination,srhFits.antennaA[vis], srhFits.antennaB[vis])#, correct_positions = False)
        self.dataTable.close()
                        
    def initAntennaTable(self, srhFits):
        self.antennaTable.addrows(48)
        
        for ant in range(32):
            self.antennaTable.col('POSITION')[ant] = (self.x, self.y - (16.5 - ant) * srhFits.RAO.base, self.z)
            self.antennaTable.col('DISH_DIAMETER')[ant] = 1.8
            self.antennaTable.col('TYPE')[ant] = 'GROUND-BASED'
            self.antennaTable.col('MOUNT')[ant] = 'EQUATORIAL'
            self.antennaTable.col('STATION')[ant] = 'WE%03d' % (ant)
            self.antennaTable.col('NAME')[ant] = 'WE%03d' % (ant)
            
        for ant in range(16):
            self.antennaTable.col('POSITION')[32 + ant] = (self.x + (ant + .5) * srhFits.RAO.base, self.y, self.z)
            self.antennaTable.col('DISH_DIAMETER')[32 + ant] = 1.8
            self.antennaTable.col('TYPE')[32 + ant] = 'GROUND-BASED'
            self.antennaTable.col('MOUNT')[32 + ant] = 'EQUATORIAL'
            self.antennaTable.col('STATION')[32 + ant] = 'S%03d' % (32 + ant)
            self.antennaTable.col('NAME')[32 + ant] = 'S%03d' % (32 + ant)
        self.antennaTable.close()
    
    def initSpectralWindowTable(self,  srhFits, frequencyChannel):
        self.spectralWindowTable.addrows()
        frequenciesAmount = 1
        freqChan = NP.zeros(frequenciesAmount)
        freqChan[:] = 1e7
        frequencies = srhFits.freqList[frequencyChannel]*1e6
        self.spectralWindowTable.col('CHAN_FREQ')[0] = frequencies
        self.spectralWindowTable.col('CHAN_WIDTH')[0] = freqChan
        self.spectralWindowTable.col('MEAS_FREQ_REF')[0] = 4
        self.spectralWindowTable.col('EFFECTIVE_BW')[0] = freqChan
        self.spectralWindowTable.col('RESOLUTION')[0] = freqChan
        self.spectralWindowTable.col('TOTAL_BANDWIDTH')[0] = 4e9
        self.spectralWindowTable.col('NUM_CHAN')[0] = frequenciesAmount
        self.spectralWindowTable.col('NET_SIDEBAND')[0] = 1
        self.spectralWindowTable.col('REF_FREQUENCY')[0] = 4e9
        self.spectralWindowTable.close()

    def initDataDescriptionTable(self):
        self.dataDescriptionTable.addrows()
        self.dataDescriptionTable.col('SPECTRAL_WINDOW_ID')[0] = 0
        self.dataDescriptionTable.col('POLARIZATION_ID')[0] = 0
        self.dataDescriptionTable.col('FLAG_ROW')[0] = 0
        self.dataDescriptionTable.close()
        
    def initPolarizationTable(self):
        self.polarizationTable.addrows()
        self.polarizationTable.col('NUM_CORR')[0] = 2
        self.polarizationTable.col('CORR_TYPE')[0] = NP.array([5, 8])
        self.polarizationTable.col('CORR_PRODUCT')[0] = NP.array([[0, 1], [0, 1]])
        self.polarizationTable.close()

    def initSourceTable(self):
        self.sourceTable.addrows()
        self.sourceTable.col('NAME')[0] = 'Sun'
        self.sourceTable.close()

    def initFieldTable(self):
        self.fieldTable.addrows()
        self.fieldTable.col('PHASE_DIR')[0] = NP.array([[1.0], [0.1]])
        self.fieldTable.col('DELAY_DIR')[0] = NP.array([[1.0], [0.1]])
        self.fieldTable.col('REFERENCE_DIR')[0] = NP.array([[1.0], [0.1]])
        self.fieldTable.col('CODE')[0] = 'S'
        self.fieldTable.col('NAME')[0] = 'Sun'
        self.fieldTable.close()

    def initFeedTable(self, srhFits, frequencyChannel):
        self.feedTable.addrows(48)
        for ant in range(48):
            self.feedTable.col('POSITION')[ant] = NP.array([0.,0.,0.])
            self.feedTable.col('BEAM_OFFSET')[ant] = NP.array([[0.],[0.]])
            self.feedTable.col('POLARIZATION_TYPE')[ant] = ['R','L']
            self.feedTable.col('POL_RESPONSE')[ant] = NP.array([[1,0],[0,1]],dtype='complex')
            self.feedTable.col('RECEPTOR_ANGLE')[ant] = NP.array([0.,0.])
            self.feedTable.col('ANTENNA_ID')[ant] = ant
            self.feedTable.col('TIME')[ant] = srhFits.freqTime[frequencyChannel, 0]
            self.feedTable.col('NUM_RECEPTORS')[ant] = 2
            self.feedTable.col('SPECTRAL_WINDOW_ID')[ant] = -1
        self.feedTable.close()

    def initObservationTable(self, srhFits, observer = 'Olga V. Melnikova'):
        self.observationTable.addrows()
        dateStart = srhFits.hduList[0].header['DATE-OBS'].replace('/','-')
        dateFinish = srhFits.hduList[0].header['DATE-END'].replace('/','-')
        timeStart = srhFits.hduList[0].header['TIME-OBS'].split('.')[0]
        timeFinish = srhFits.hduList[0].header['TIME-END'].split('.')[0]
        
        t_start = Time(dateStart + ' ' + timeStart, scale='utc')
        t_finish = Time(dateFinish + ' ' + timeFinish, scale='utc')
        self.observationTable.col('TELESCOPE_NAME')[0] = 'SRH48'
        self.observationTable.col('OBSERVER')[0] = observer
        self.observationTable.col('RELEASE_DATE')[0] = t_start.jd1
        self.observationTable.col('TIME_RANGE')[0] = [t_start.jd1,t_finish.jd1]
        self.observationTable.close()
        
    def antennaName2Index(self, ant):
        ind = -1
        if ant >=49 and ant <=80:
            ind = ant - 49
        if ant >= 176 and ant <=192:
            ind = 192 - ant + 32
        return ind
            