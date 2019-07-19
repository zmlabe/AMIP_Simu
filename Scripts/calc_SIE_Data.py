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
year1 = 1979
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
    sic = data.variables['SIC'][12:]
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

### Detrend data
extdt = sss.detrend(ext,axis=0,type='linear')

### Calculate zscores
extdtzz = sts.zscore(extdt,axis=0)
extzz = sts.zscore(ext,axis=0)

### Calculate standard deviation
extstd = np.std(ext,axis=0)

### Calculate OND sea ice index
extond = np.nanmean(ext[:,-3:],axis=1)
extondzz = sts.zscore(extond,axis=0)
iceslice05_ond = np.where(extondzz <= -0.5)[0]
yearslice05_ond = years[iceslice05_ond]
iceslice1_ond = np.where(extondzz <= -1.)[0]
yearslice1_ond = years[iceslice1_ond]
iceslice2_ond = np.where(extondzz <= -2.)[0]
yearslice2_ond = years[iceslice2_ond]

### Calculate D sea ice index
extd = np.nanmean(ext[:,-1:],axis=1)
extdzz = sts.zscore(extd,axis=0)
iceslice05_d = np.where(extdzz <= -0.5)[0]
yearslice05_d = years[iceslice05_d]
iceslice1_d = np.where(extdzz <= -1.)[0]
yearslice1_d = years[iceslice1_d]
iceslice2_d = np.where(extdzz <= -2.)[0]
yearslice2_d = years[iceslice2_d]

### Calculate JFM sea ice index
extjfm = np.nanmean(ext[:,0:3],axis=1)
extjfmzz = sts.zscore(extjfm,axis=0)
iceslice05_jfm = np.where(extjfmzz <= -0.5)[0]
yearslice05_jfm = years[iceslice05_jfm]
iceslice1_jfm = np.where(extjfmzz <= -1.)[0]
yearslice1_jfm = years[iceslice1_jfm]
iceslice2_jfm = np.where(extjfmzz <= -2.)[0]
yearslice2_jfm = years[iceslice2_jfm]

### Save data files
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent.txt',
           ext.ravel())
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent_ZScore.txt',
           extzz.ravel())
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent_ZScore_Detrended.txt',
           extdtzz.ravel())
np.savetxt(directoryoutput + 'Monthly_SeaIceExtent_STD.txt',
           extstd.ravel())

### Save OND file
np.savetxt(directoryoutput + 'OND_SeaIceExtent.txt',
           extond)
np.savetxt(directoryoutput + 'OND_SeaIceExtent_ZScore.txt',
           extondzz)
np.savetxt(directoryoutput + 'OND_SeaIceExtent_05SigmaYears.txt',
           np.c_[yearslice05_ond,iceslice05_ond])
np.savetxt(directoryoutput + 'OND_SeaIceExtent_1SigmaYears.txt',
           np.c_[yearslice1_ond,iceslice1_ond])
np.savetxt(directoryoutput + 'OND_SeaIceExtent_2SigmaYears.txt',
           np.c_[yearslice2_ond,iceslice2_ond])

### Save D file
np.savetxt(directoryoutput + 'D_SeaIceExtent.txt',
           extd)
np.savetxt(directoryoutput + 'D_SeaIceExtent_ZScore.txt',
           extdzz)
np.savetxt(directoryoutput + 'D_SeaIceExtent_05SigmaYears.txt',
           np.c_[yearslice05_d,iceslice05_d])
np.savetxt(directoryoutput + 'D_SeaIceExtent_1SigmaYears.txt',
           np.c_[yearslice1_d,iceslice1_d])
np.savetxt(directoryoutput + 'D_SeaIceExtent_2SigmaYears.txt',
           np.c_[yearslice2_d,iceslice2_d])

### Save JFM file
np.savetxt(directoryoutput + 'JFM_SeaIceExtent.txt',
           extjfm)
np.savetxt(directoryoutput + 'JFM_SeaIceExtent_ZScore.txt',
           extjfmzz)
np.savetxt(directoryoutput + 'JFM_SeaIceExtent_05SigmaYears.txt',
           np.c_[yearslice05_jfm,iceslice05_jfm])
np.savetxt(directoryoutput + 'JFM_SeaIceExtent_1SigmaYears.txt',
           np.c_[yearslice1_jfm,iceslice1_jfm])
np.savetxt(directoryoutput + 'JFM_SeaIceExtent_2SigmaYears.txt',
           np.c_[yearslice2_jfm,iceslice2_jfm])

### Create temporary figures
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

sic1 = np.reshape(sic,(456//12,12,lat.shape[0],lon.shape[0]))
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

fig = plt.figure()
plt.plot(years,extondzz,marker='o',markersize=7,
         linewidth=5,color='forestgreen',clip_on=False)
plt.axhline(-0.5,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.axhline(-1,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.axhline(-2,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.xlim([1979,2016])
plt.savefig(directoryfigure + 'ONDSIE_zscore.png',dpi=300)

fig = plt.figure()
plt.plot(years,extdzz,marker='o',markersize=7,
         linewidth=5,color='forestgreen',clip_on=False)
plt.axhline(-0.5,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.axhline(-1,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.axhline(-2,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.xlim([1979,2016])
plt.savefig(directoryfigure + 'DSIE_zscore.png',dpi=300)

fig = plt.figure()
plt.plot(years,extjfmzz,marker='o',markersize=7,
         linewidth=5,color='forestgreen',clip_on=False)
plt.axhline(-0.5,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.axhline(-1,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.axhline(-2,color='dimgrey',linestyle='--',dashes=(1,0.3),linewidth=2)
plt.xlim([1979,2016])
plt.savefig(directoryfigure + 'DSIE_zscore.png',dpi=300)