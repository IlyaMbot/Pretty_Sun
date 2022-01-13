import sys
sys.path.append('./srhedik')
from srhedik import srhMS2
from srhFitsFile_doubleBase import SrhFitsFile

filename = "srh_20210603T035451.fit"

saveName = "test"
currentFrequencyChannel = 0

uvSize = 512
srhFits = SrhFitsFile( filename , uvSize)



try:
    ms2Table = srhMS2.SrhMs2(saveName)
    ms2Table.initDataTable(self.srhFits, currentFrequencyChannel, self.ewLcpPhaseCorrection, self.ewRcpPhaseCorrection, self.sLcpPhaseCorrection, self.sRcpPhaseCorrection, phaseCorrect = self.phaseCorrect, amplitudeCorrect = self.amplitudeCorrect)
    ms2Table.initAntennaTable(self.srhFits)
    ms2Table.initSpectralWindowTable(self.srhFits, currentFrequencyChannel)
    ms2Table.initDataDescriptionTable()
    ms2Table.initPolarizationTable()
    ms2Table.initSourceTable()
    ms2Table.initFieldTable()
    ms2Table.initFeedTable(self.srhFits, currentFrequencyChannel)
    ms2Table.initObservationTable(self.srhFits)
except:
    print("SOMETHING'S WRONG")