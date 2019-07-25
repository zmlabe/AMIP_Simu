"""
Script calculates Eurasian snow area index for October-November using data
from the Rutgers Global Snow Lab data

Notes
-----
    Author : Zachary Labe
    Date   : 25 July 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.signal as SS

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
yearsdata = np.arange(year1,2018+1,1)
m = 12 # number of months

### Read in all months of data
yearsdata,months,data = np.genfromtxt(directoryoutput + \
                                  'CDR_SCE_Eurasia_Monthly.txt',unpack=True,
                                  usecols=[0,1,2])

### Reshape data into []
yearssort = np.reshape(yearsdata,(yearsdata.shape[0]//m,m))
monthsort = np.reshape(months,(months.shape[0]//m,m))
datasortq = np.reshape(data,(data.shape[0]//m,m))

### Change units from km^2 to 10^6 km^2
datasort = datasortq/1e6

### Calculate October-November index (1979-2015)
octnov = np.nanmean(datasort[:years.shape[0],9:11],axis=1)
octnovdt = SS.detrend(octnov,type='linear')

### Calculate October index (1979-2015)
octonly = datasort[:years.shape[0],9:10].squeeze()
octonlydt = SS.detrend(octonly,type='linear')

### Save both indices (Oct-Nov)
np.savetxt(directoryoutput + 'SNA_Eurasia_ON_CDRSCE.txt',
           np.vstack([years,octnov]).transpose(),delimiter=',',fmt='%3.1f',
           footer='\n Snow cover index calculated from' \
           'CDR SCE record in Global Snow Lab by \n' \
           'Rutgers',newline='\n\n')
np.savetxt(directoryoutput + 'SNA_Eurasia_ON_CDRSCE_DETRENDED.txt',
           np.vstack([years,octnovdt]).transpose(),delimiter=',',fmt='%3.1f',
           footer='\n Snow cover index calculated from' \
           'CDR SCE record in Global Snow Lab by \n' \
           'Rutgers',newline='\n\n')

### Save both indices (Oct)
np.savetxt(directoryoutput + 'SNA_Eurasia_O_CDRSCE.txt',
           np.vstack([years,octonly]).transpose(),delimiter=',',fmt='%3.1f',
           footer='\n Snow cover index calculated from' \
           'CDR SCE record in Global Snow Lab by \n' \
           'Rutgers',newline='\n\n')
np.savetxt(directoryoutput + 'SNA_Eurasia_O_CDRSCE_DETRENDED.txt',
           np.vstack([years,octonlydt]).transpose(),delimiter=',',fmt='%3.1f',
           footer='\n Snow cover index calculated from' \
           'CDR SCE record in Global Snow Lab by \n' \
           'Rutgers',newline='\n\n')