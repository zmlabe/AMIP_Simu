"""
Script calculates Eurasian snow index for October-November following the 
methods of Peings et al. 2017

Notes
-----
    Author : Zachary Labe
    Date   : 22 July 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
import read_MonthlyData as MOM
import calc_Utilities as UT
import scipy.stats as sts
import scipy.signal as SS
import read_Reanalysis as MOR

### Define directories
directoryfigure = '/home/zlabe/Desktop/'
directoryoutput = '/home/zlabe/Documents/Research/AMIP/Data/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Calculating Snow Cover Index - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2015
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
varnames = 'SNC'
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readVar(varnames,runnamesm):
### Call function to read in ERA-Interim
    lat,lon,time,lev,era = MOR.readDataR('T2M','surface',False,True)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                   'surface',False,True)
        
    ### Select October-November for index
    modq = np.nanmean(models[:,:,:,9:11,:,:],axis=3)
    
    ### Calculate ensemble mean
    modmean = np.nanmean(modq[:,:,:,:,:],axis=1)
            
    return modmean,lat,lon,lev

###############################################################################
### Read in data functions
mod,lat,lon,lev = readVar(varnames,runnamesm)

### Slice over region of interest for Eurasia (40-80N,35-180E)
latq = np.where((lat >= 40) & (lat <= 80))[0]
lonq = np.where((lon >= 35) & (lon <=180))[0]
latn = lat[latq]
lonn = lon[lonq]
lon2,lat2 = np.meshgrid(lonn,latn)

modlat = mod[:,:,latq,:]
modlon = modlat[:,:,:,lonq]
modslice = modlon.copy()

### Consider years 1979-2015
modsliceq = modslice[:,:-1]

### Calculate average snow index
snowindex = UT.calc_weightedAve(modsliceq,lat2)

### Calculate detrended snow index
snowindexdt = SS.detrend(snowindex,type='linear',axis=1)

### Save both indices
np.savetxt(directoryoutput + 'SNCI_Eurasia_ON.txt',
           np.vstack([years,snowindex]).transpose(),delimiter=',',fmt='%3.1f',
           footer='\n Snow cover index calculated for each' \
           '\n experiment [CSST,CSIC,AMIP,AMQ,AMS,AMQS]\n' \
           ' in Oct-Nov (not SWE)',newline='\n\n')
np.savetxt(directoryoutput + 'SNCI_Eurasia_ON_DETRENDED.txt',
           np.vstack([years,snowindexdt]).transpose(),delimiter=',',fmt='%3.1f',
           footer='\n Snow cover index calculated for each' \
           '\n experiment [CSST,CSIC,AMIP,AMQ,AMS,AMQS]\n' \
           ' in Oct-Nov ---> detrended data (not SWE)',newline='\n\n')