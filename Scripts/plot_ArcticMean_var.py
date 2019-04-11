"""
Script plots mean variables over the polar cap between the WACCM experiments
and ERA-Interim. 

Notes
-----
    Author : Zachary Labe
    Date   : 20 February 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import read_MonthlyData as MOM
import read_Reanalysis as MOR
import calc_Utilities as UT
import scipy.stats as sts

### Define directories
directoryfigure = '/home/zlabe/Desktop/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting WACC T2M Trends - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
varnames = ['T2M']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readVar(varnames,slicegrid,region):
    ### Call function to read in ERA-Interim
    lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',False,True)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                   'surface',False,True)
        
    ### Retrieve time period of interest
    modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                       era.shape[3]))
    for i in range(len(runnamesm)):
        for j in range(ensembles):
            modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                                lat,lon,'surface',1)
    eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    
    ### Take ensemble mean
    modmean = np.nanmean(modq,axis=1)
    
    ### Slice over the polar cap
    if slicegrid == True:
        if region == 'Siberia':
            latslicemin = 40
            latslicemax = 65
            lonslicemin = 50
            lonslicemax = 130
            latq = np.where((lat >= latslicemin) & (lat <= latslicemax))[0]
            lat = lat[latq]
            lonq = np.where((lon >= lonslicemin) & (lon <= lonslicemax))[0]
            lon = lon[lonq]
            eraq = eraq[:,latq,:]
            eraq = eraq[:,:,lonq]
            modmean = modmean[:,:,latq,:]
            modmean = modmean[:,:,:,lonq]
        elif region == 'NA':
            latslicemin = 35
            latslicemax = 50
            lonslicemin = 260
            lonslicemax = 290
            latq = np.where((lat >= latslicemin) & (lat <= latslicemax))[0]
            lat = lat[latq]
            lonq = np.where((lon >= lonslicemin) & (lon <= lonslicemax))[0]
            lon = lon[lonq]
            eraq = eraq[:,latq,:]
            eraq = eraq[:,:,lonq]
            modmean = modmean[:,:,latq,:]
            modmean = modmean[:,:,:,lonq]
        elif region == 'Arctic':
            latq = np.where(lat >= 65)[0]
            lat = lat[latq]
            eraq = eraq[:,latq,:]
            modmean = modmean[:,:,latq,:]
        elif region == 'NH':
            eraq = eraq 
            modmean = modmean
    
    ### Meshgrid for lat/lon
    lon2,lat2 = np.meshgrid(lon,lat)
    
    eramean = UT.calc_weightedAve(eraq,lat2)
    mmean = UT.calc_weightedAve(modmean,lat2)
    
    ### Create climo over time series
    eraiave = np.nanmean(eramean)
    modelave = np.nanmean(mmean,axis=1)
    
    ### Calculate anomalies
    eraanom = eramean - eraiave
    modelanom = (mmean.transpose() - modelave).transpose()
    
    return eraanom, modelanom

### Read in data
region = 'Arctic'
eraanom,modelanom = readVar(varnames[0],True,region)
    
corrs = np.empty((modelanom.shape[0]))
for i in range(modelanom.shape[0]):
    corrs[i] = sts.pearsonr(eraanom,modelanom[i])[0]
    
############################################################################
############################################################################
############################################################################
##### Plot mean values over the polar cap
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

fig = plt.figure()
ax = plt.subplot(111)

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 2))
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

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',pad=1)

color=iter(cmocean.cm.thermal(np.linspace(0.,1,len(modelanom)+1)))
for i in range(len(modelanom)+1):
    c=next(color)
    if i == 6:
        plt.plot(eraanom,linewidth=3,color='k',alpha=1,
             label = r'\textbf{%s}' % 'ERA-I',linestyle='-',
             marker='o',markersize=4,clip_on=False,zorder=10)
    else:
        plt.plot(modelanom[i],linewidth=2,color=c,alpha=1,
                 label = r'\textbf{%s}(%s)' % (runnamesm[i],
                                  np.round(corrs[i],2)),
                                  linestyle='-',marker='o',
                                  markersize=0,clip_on=False)

plt.legend(shadow=False,fontsize=8,loc='upper center',
           fancybox=True,frameon=False,ncol=4,bbox_to_anchor=(0.5, 1.03),
           labelspacing=0.2,columnspacing=1,handletextpad=0.4,
           edgecolor='dimgrey') 

plt.ylabel(r'\textbf{T2M ($^{\circ}$C)}',color='k',fontsize=10)
plt.yticks(np.arange(-6,7,1),list(map(str,np.arange(-6,7,1))),fontsize=6)
plt.ylim([-4,4])

plt.xticks(np.arange(0,40,3),np.arange(1980,2019,3),fontsize=6)
plt.xlim([0,37])

plt.savefig(directoryfigure + 'mean_T2M_%s.png' % region,dpi=300)