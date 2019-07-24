"""
Script calculates Eurasian snow area index for October-November following the 
methods of Peings et al. 2017 in ERA-Interim (land)

Notes
-----
    Author : Zachary Labe
    Date   : 24 July 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
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
print('\n' '----Calculating Snow Cover Area Index - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2015
years = np.arange(year1,year2+1,1)

### Add parameters
varnames = 'SNC'

def readVar(varnames):
    ### Call function to read in ERA-Interim
    lat,lon,time,lev,era = MOR.readDataR('SNC','surface',False,True)
        
    ### Select October-November for index
    eraq = np.nanmean(era[:,9:11,:,:],axis=1)
#    eraq = era[:,9:10,:,:].squeeze() # Octobers
            
    return eraq,lat,lon,lev

###############################################################################
### Read in data functions
era,lat,lon,lev = readVar(varnames)

### Slice over region of interest for Eurasia (40-80N,35-180E)
latq = np.where((lat >= 40) & (lat <= 80))[0]
lonq = np.where((lon >= 35) & (lon <=180))[0]
latn = lat[latq]
lonn = lon[lonq]
lon2,lat2 = np.meshgrid(lonn,latn)

eralat = era[:,latq,:]
eralon = eralat[:,:,lonq]
eraslice = eralon.copy()

### Calculate sea ice extent
def calcExtent(snowq,lat2):
    """
    Calculate snow cover extent from snow concentration grids following
    the methods of Robinson et al. 1993 [BAMS]
    """
    ### Extent is a binary 0 or 1 for 50% snow threshold
    thresh=50.
    snow = snowq.copy()
    snow[np.where(snow<thresh)]=np.nan
    snow[np.where(snow>thresh)]=1
    
    ext = np.zeros((snow.shape[0]))
    valyr = np.zeros((snow.shape))
    for ti in range(snow.shape[0]):
        for i in range(snow.shape[1]):
            for j in range(snow.shape[2]):
                if snow[ti,i,j] == 1.0:
                   ### Area 1.9x2.5 grid cell [58466.1 = (278.30) * (210.083)]
                   valyr[ti,i,j] = 58466.1 * np.cos(np.radians(lat2[i,j]))
        ext[ti] = np.nansum(valyr[ti,:,:])/1e6
        
    return ext

### Calculate snow cover area
snowarea = calcExtent(eraslice,lat2)

### Calculate detrended snow index
snowareaindexdt = SS.detrend(snowarea,type='linear')

#### Save both indices
#np.savetxt(directoryoutput + 'SNA_Eurasia_ON_ERAi.txt',
#           np.vstack([years,snowarea]).transpose(),delimiter=',',fmt='%3.1f',
#           footer='\n Snow cover index calculated for the' \
#           '\n ERA-Interim reanalysis from 1979-2015\n' \
#           ' in Oct-Nov (AREA)',newline='\n\n')
#np.savetxt(directoryoutput + 'SNA_Eurasia_ON_ERAi_DETRENDED.txt',
#           np.vstack([years,snowareaindexdt]).transpose(),delimiter=',',fmt='%3.1f',
#           footer='\n Snow cover index calculated for the' \
#           '\n ERA-Interim reanalysis from 1979-2015\n' \
#           ' in Oct-Nov ---> detrended data (AREA)',newline='\n\n')