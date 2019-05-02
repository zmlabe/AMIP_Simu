"""
Script calculates and plots climatologies of 10 hPa zonal wind at 65N for 
model experiments and ERAi.

Notes
-----
    Author : Zachary Labe
    Date   : 2 May 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
import read_MonthlyData as MOM
import read_Reanalysis as MOR
import calc_Utilities as UT
import scipy.stats as sts
import scipy.signal as sss
from netCDF4 import Dataset

### Define directories
directoryfigure = '/home/zlabe/Desktop/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting U10 Variability in Polar Vortex- %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
period = 'DJF'
varnames = ['U10']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readVar(varnames,runnamesm,period):
    ### Call function to read in ERA-Interim (detrended)
    lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',True,True)
    
    ### Call functions to read in WACCM data (detrended)
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                   'surface',True,True)
        
    ### Retrieve time period of interest
    if period == 'ON':
        modq = np.nanmean(models[:,:,:,9:11,:,:],axis=3)
        eraq = np.nanmean(era[:,9:11,:,:],axis=1)
    elif period == 'OND':
        modq = np.nanmean(models[:,:,:,-3:,:,:],axis=3)
        eraq = np.nanmean(era[:,-3:,:,:],axis=1)
    elif period == 'ND':
        modq = np.nanmean(models[:,:,:,-2:,:,:],axis=3)
        eraq = np.nanmean(era[:,-2:,:,:],axis=1)
    elif period == 'N':
        modq = models[:,:,:,-2,:,:].squeeze()
        eraq = era[:,-2,:,:].squeeze()
    elif period == 'D':
        modq = models[:,:,:,-1:,:,:].squeeze()
        eraq = era[:,-1:,:,:].squeeze()
    elif period == 'JJA':
        modq = np.nanmean(models[:,:,:,5:8,:,:],axis=3)
        eraq = np.nanmean(era[:,5:8,:,:],axis=1)  
    elif period == 'Annual':
        modq = np.nanmean(models[:,:,:,:,:,:],axis=3)
        eraq = np.nanmean(era[:,:,:,:],axis=1)  
    elif period == 'DJF':
        modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                           era.shape[3]))
        for i in range(len(runnamesm)):
            for j in range(ensembles):
                modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:,:],
                                                    lat,lon,'surface',1)
        eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
     
    ### Slice U10 at 65N
    latq = np.where((lat >= 64.5) & (lat <= 65.5))[0]
    lat = lat[latq].squeeze()
    eraq = eraq[:,latq,:].squeeze()
    modq = modq[:,:,:,latq,:].squeeze()
    
    ### Take zonal mean
    eraq = np.nanmean(eraq,axis=1)
    modq = np.nanmean(modq,axis=3)
    
    return modq,eraq,lat,lon

def read300yr(period):
    """
    Read in 300 year control
    """
    directory300 = '/seley/ypeings/simu/PAMIP-1.1-QBO-300yr/monthly/'
    file300 = 'U10_1700-2000.nc'
    filename = directory300 + file300
    
    data = Dataset(filename)
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    u10q = data.variables['U10'][:]
    data.close()
    
    ### Reshape in year/month
    u10n = np.reshape(u10q,(u10q.shape[0]//12,12,lat.shape[0],lon.shape[0]))
    
    ### Calculate over particular months
    u10 = UT.calcDecJanFeb(u10n,lat,lon,'surface',1)
    
    ### Slice U10 at 65N
    latq = np.where((lat >= 64.5) & (lat <= 65.5))[0]
    lat = lat[latq].squeeze()
    u10 = u10[:,latq,:].squeeze()
    
    ### Take zonal mean 
    u10z = np.nanmean(u10,axis=1)
    
    ### Remove missing data
    mask = np.where(u10z > -1e5)[0]
    
    ### Detrend
    u10zdt = sss.detrend(u10z[mask],type='linear')
    
    return lat,lon,u10zdt

def read300yrh(period):
    """
    Read in 300 year control
    """
    directory300 = '/seley/ypeings/simu/PAMIP-1.1-QBO-300yr/monthly/'
    file300 = 'U10_1700-2000.nc'
    filename = directory300 + file300
    
    data = Dataset(filename)
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    u10q = data.variables['U10'][:]
    data.close()
    
    ### Reshape in year/month
    u10n = np.reshape(u10q,(u10q.shape[0]//12,12,lat.shape[0],lon.shape[0]))
    
    ### Calculate over particular months
    u10 = UT.calcDecJanFeb(u10n,lat,lon,'surface',1)
    
    ### Slice U10 at 65N
    latq = np.where((lat >= 64.5) & (lat <= 65.5))[0]
    lat = lat[latq].squeeze()
    u10 = u10[:,latq,:].squeeze()
    
    ### Take zonal mean 
    u10z = np.nanmean(u10,axis=1)
    
    ### Remove missing data
    mask = np.where(u10z > -1e5)[0]
    
    ### Detrend
    u10zdt = sss.detrend(u10z[mask],type='linear')
    
    return lat,lon,u10zdt

def read300yrf(period):
    """
    Read in 300 year control
    """
    directory300 = '/seley/ypeings/simu/PAMIP-1.6-QBO-300yr/monthly/'
    file300 = 'U10_1700-2000.nc'
    filename = directory300 + file300
    
    data = Dataset(filename)
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    u10q = data.variables['U10'][:]
    data.close()
    
    ### Reshape in year/month
    u10n = np.reshape(u10q,(u10q.shape[0]//12,12,lat.shape[0],lon.shape[0]))
    
    ### Calculate over particular months
    u10 = UT.calcDecJanFeb(u10n,lat,lon,'surface',1)
    
    ### Slice U10 at 65N
    latq = np.where((lat >= 64.5) & (lat <= 65.5))[0]
    lat = lat[latq].squeeze()
    u10 = u10[:,latq,:].squeeze()
    
    ### Take zonal mean 
    u10z = np.nanmean(u10,axis=1)
    
    ### Remove missing data
    mask = np.where(u10z > -1e5)[0]
    
    ### Detrend
    u10zdt = sss.detrend(u10z[mask],type='linear')
    
    return lat,lon,u10zdt

### Read in data
#mod,era,lat,lon = readVar(varnames[0],runnamesm,period) 
#lat,lon,u10h = read300yrh(period)
#lat,lon,u10f = read300yrf(period)
    
### Plot all experiments as an ensemble member
modens = np.reshape(mod,(mod.shape[0]*mod.shape[1],mod.shape[2]))

### Ensemble mean
modmean = np.nanmean(mod,axis=1)

### Correlations
corrs = np.empty((modmean.shape[0])) 
pval = np.empty((modmean.shape[0])) 
for i in range(modmean.shape[0]):
    corrs[i],pval[i] = sts.pearsonr(era,modmean[i,:])
    print('Correlation (r) %s for ERAi and %s' % (np.round(corrs[i],2),
                                                  runnamesm[i]))

corrsmem = np.empty((mod.shape[0],mod.shape[1]))    
for i in range(mod.shape[0]):
    for j in range(mod.shape[1]):
        corrsmem[i,j] = sts.pearsonr(era,mod[i,j,:])[0]
    
### Standard Deviation
modstd = np.empty((mod.shape[0],mod.shape[1])) 
for i in range(mod.shape[0]):
    for j in range(mod.shape[1]):
        modstd[i,j] = np.nanstd(mod[i,j,:])
        
modmstd = np.nanmean(modstd,axis=1)
for i in range(mod.shape[0]):
    print('Standard Deviation is %s for %s' % (np.round(modmstd[i],2),
                                                runnamesm[i]))
erastd = np.nanstd(era)
print('Standard Deviation is %s for ERAi' % np.round(erastd,2))
    
###########################################################################
###########################################################################
###########################################################################
##### Plot profiles
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([]) 
        
### Create plot
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')

plt.plot(modens.transpose(),linestyle='-',color='dimgrey',
         linewidth=0.8,alpha=0.45)
plt.plot(era,linestyle='-',color='k',linewidth=4,label=r'ERAi',
         clip_on=False)

plt.ylabel(r'\textbf{U10 [m/s]}',color='k',fontsize=12)
plt.xticks(np.arange(0,38,5),map(str,np.arange(1980,2017,5)))
plt.xlim([0,37])

plt.yticks(np.arange(-80,81,10),map(str,np.arange(-80,81,10)))
plt.ylim([-30,30])

plt.savefig(directoryfigure + 'U10_climo_%s.png' % period,dpi=300)

################################################################################
#### Create plot
#fig = plt.figure()
#ax = plt.subplot(111)
#adjust_spines(ax, ['left', 'bottom'])
#ax.spines['top'].set_color('none')
#ax.spines['right'].set_color('none')
#ax.spines['left'].set_color('dimgrey')
#ax.spines['bottom'].set_color('dimgrey')
#ax.spines['left'].set_linewidth(2)
#ax.spines['bottom'].set_linewidth(2)
#ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
#
#plt.plot(mod[5].transpose(),linestyle='-',color='dimgrey',
#         linewidth=0.8,alpha=1)
#plt.plot(era,linestyle='-',color='k',linewidth=4,label=r'ERAi',
#         clip_on=False)
#
#plt.ylabel(r'\textbf{U10 [m/s]}',color='k',fontsize=12)
#plt.xticks(np.arange(0,38,5),map(str,np.arange(1980,2017,5)))
#plt.xlim([0,37])
#
#plt.yticks(np.arange(-80,81,10),map(str,np.arange(-80,81,10)))
#plt.ylim([-30,30])
#
#plt.savefig(directoryfigure + 'U10_climo_%s_AMQS.png' % period,dpi=300)

### Create plot
fig = plt.figure(figsize=(10,3))
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')

plt.plot(u10h,linestyle='-',color='k',linewidth=2,label=r'300-yr h',
         clip_on=False)
plt.plot(u10f,linestyle='-',color='m',linewidth=2,label=r'300-yr f',
         clip_on=False)
plt.plot(mod.ravel(),linestyle='-',color='deepskyblue',linewidth=0.5,
         label=r'AMIPs',clip_on=False)
plt.plot(era,linestyle='-',color='gold',linewidth=1,label=r'ERAi',
         clip_on=False)

plt.ylabel(r'\textbf{U10 [m/s]}',color='k',fontsize=12)
#plt.xticks(np.arange(0,38,5),map(str,np.arange(1980,2017,5)))
plt.xlim([0,1000])

plt.yticks(np.arange(-80,81,10),map(str,np.arange(-80,81,10)))
plt.ylim([-30,30])

plt.savefig(directoryfigure + 'U10_climo_%s.png' % period,dpi=300)