"""
Script calculates sea ice extent and other various statistics. Several output
files are also created.

Notes
-----
    Author : Zachary Labe
    Date   : 12 March 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import scipy.stats as sts
import scipy.signal as sss

### Define directories
directoryfigure = '/home/zlabe/Desktop/SIE/'
directoryoutput = '/home/zlabe/Documents/Research/AMIP/Data/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Calculating Sea Ice Extent Data - %s----' % titletime)

#### Alott time series
year1 = 1978
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 1

### Function to read in data
def readSIEData():
    """ 
    Read data from AMIP1 and use land mask
    """
    directorydata = '/seley/zlabe/simu/AMIP1/monthly/'
    filename = 'SIC_1978-2016.nc'
    
    data = Dataset(directorydata + filename)
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    sic = data.variables['SIC'][:]
    data.close()
    
    ### Mask data 
    directorymask = '/seley/zlabe/simu/masks/'
    filenamemask = 'domain.camocn.1.9x2.5_gx1v6_090403.nc'
    datam = Dataset(directorymask + filenamemask)
    mask = datam.variables['frac'][:]
    datam.close()
    
    ### Set missing data
    sic = sic * mask
    sic[sic<0] = 0
    sic[sic>100] = 100
    
    ### Slice for Arctic data (Northern Hemisphere)
    latq = np.where(lat >= 0)[0]
    lat = lat[latq]
    sic = sic[:,latq,:]
    
    return lat,lon,sic

### Calculate sea ice extent
def calcExtent(sic,lat2):
    """
    Calculate sea ice extent from sea ice concentration grids
    """
    ### Extent is a binary 0 or 1 for 15% SIC threshold
    thresh=15
    sic[np.where(sic<thresh)]=np.nan
    sic[np.where(sic>=thresh)]=1
    
    ext = np.zeros((sic.shape[0]))
    valyr = np.zeros((sic.shape))
    for ti in range(ext.shape[0]):
        for i in range(lat.shape[0]):
            for j in range(lon.shape[0]):
                if sic[ti,i,j] == 1.0:
                   ### Area 1.9x2.5 grid cell [58466.1 = (278.30) * (210.083)]
                   valyr[ti,i,j] = 58466.1 * np.cos(np.radians(lat2[i,j]))
        ext[ti] = np.nansum(valyr[ti,:,:])/1e6
        
    ### Reshape array for [year,month]
    ext = np.reshape(ext,(ext.shape[0]//12,12))
    return ext

### Calculate functions
lat,lon,sic = readSIEData()
lon2,lat2 = np.meshgrid(lon,lat)
ext = calcExtent(sic,lat2)

### Calculate zscores
extdt = sss.detrend(ext,axis=0,type='linear')
extdtzz = sts.zscore(extdt,axis=0)
extzz = sts.zscore(ext,axis=0)
extstd = np.std(ext,axis=0)

### Save data files
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent.txt',
           ext.ravel())
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent_ZScore.txt',
           extzz.ravel())
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent_ZScore_Detrended.txt',
           extdtzz.ravel())
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent_STD.txt',
           extstd.ravel())

### Create temporary figures
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

sic1 = np.reshape(sic,(468//12,12,lat.shape[0],lon.shape[0]))
fig = plt.figure()
m = Basemap(projection='npstere',boundinglat=30,lon_0=0,
                                resolution='l',round =True,area_thresh=10000.)
cs = m.contourf(lon2,lat2,sic1[-5,3,:,:],50,extend='both',latlon=True)
m.drawcoastlines(color='darkgray',linewidth=0.3)
plt.colorbar()
plt.savefig(directoryfigure + 'sictest.png',dpi=300)

fig = plt.figure()
plt.plot(ext.ravel(),marker='o',markersize=1,
         linewidth=2,color='forestgreen',clip_on=False)
plt.savefig(directoryfigure + 'monthlycycle.png',dpi=300)

fig = plt.figure()
plt.plot(np.nanmean(ext,axis=0),marker='o',markersize=7,
         linewidth=5,color='forestgreen',clip_on=False)
plt.savefig(directoryfigure + 'seasonalcycle.png',dpi=300)

fig = plt.figure()
plt.plot(years,np.nanmean(ext,axis=1),marker='o',markersize=7,
         linewidth=5,color='forestgreen',clip_on=False)
plt.xlim([1978,2016])
plt.savefig(directoryfigure + 'annualSIE.png',dpi=300)

fig = plt.figure()
plt.plot(years,ext[:,8],marker='o',markersize=7,
         linewidth=5,color='forestgreen')
plt.xlim([1978,2016])
plt.savefig(directoryfigure + 'SeptemberSIE.png',dpi=300)

fig = plt.figure()
plt.plot(years,np.nanmean(ext[:,-3:],axis=1),marker='o',markersize=7,
         linewidth=5,color='forestgreen',clip_on=False)
plt.xlim([1978,2016])
plt.savefig(directoryfigure + 'ONDSIE.png',dpi=300)