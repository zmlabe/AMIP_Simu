"""
Script plots mean trends over selected regions and time periods

Notes
-----
    Author : Zachary Labe
    Date   : 28 February 2019
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
su = [0,1,2,3,5,6,7]
yearmn = 1991
yearmx = 2014
region = 'Siberia'
varnames = ['T2M','SLP','Z500','Z50','U200','U10']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readDataTrends(varnames,month,years,yearmn,yearmx,region):
    ### Call function to read in ERA-Interim
    lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',False,True)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                   'surface',False,True)
        
    ### Retrieve time period of interest
    if month == 'DJF':
        modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                           era.shape[3]))
        for i in range(len(runnamesm)):
            for j in range(ensembles):
                modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                                    lat,lon,'surface',1)
        eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    
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
        modq = modq[:,:,:,latq,:]
        modq = modq[:,:,:,:,lonq]
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
        modq = modq[:,:,:,latq,:]
        modq = modq[:,:,:,:,lonq]
    elif region == 'Arctic':
        latq = np.where(lat >= 65)[0]
        lat = lat[latq]
        eraq = eraq[:,latq,:]
        modq = modq[:,:,:,latq,:]
    elif region == 'NH':
        eraq = eraq 
        modmean = modq
    
    ### Calculate the trend for WACCM  
    modtrend = np.empty((len(runnamesm),ensembles,modq.shape[3],
                         modq.shape[4]))
    for i in range(len(runnamesm)):
        modtrend[i,:,:,:] = UT.detrendData(modq[i],years,'surface',
                                            yearmn,yearmx)
        print('Completed: Simulation --> %s!' % runnamesm[i])
        
    ### Calculate decadal trend
    dectrend = modtrend * 10.
        
    ### Calculate the trend for ERA-Interim
    retrend,std_err = UT.detrendDataR(eraq,years,'surface',yearmn,yearmx)
    
    #### Calculate decadal trend
    redectrend = retrend * 10.
    std_err = std_err * 10.
    
    return lat,lon,dectrend,redectrend,std_err
    
### Read in trends
#lat,lon,mods,eras,std_err = readDataTrends('T2M','DJF',years,yearmn,
#                                            yearmx,region)
#
#### Meshgrid of lat/lon
#lon2,lat2 = np.meshgrid(lon,lat)
#
#### Calculate average over region
#modmeans = UT.calc_weightedAve(mods,lat2)
#erameans = UT.calc_weightedAve(eras,lat2)
#errormeans = UT.calc_weightedAve(std_err,lat2)
#
#### Calculate percentiles
#perc05 = np.percentile(modmeans,5,axis=1)
#perc95 = np.percentile(modmeans,95,axis=1)
#mean = np.nanmean(modmeans,axis=1)
#
#### Calculate confidence interval for ERA-Interim
#err = 1.96*errormeans #95% level
#
#### Array of data
#p5 = np.append(np.array([-err]),perc05)
#p95 = np.append(np.array([err]),perc95)
#mean = np.append(erameans,mean)
    
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
        
xx = np.arange(len(mean))

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(0)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',pad=1)
plt.gca().axes.get_xaxis().set_visible(False)

plt.axhline(0,color='dimgrey',linewidth=2,linestyle='--',dashes=(1,0.3),
            zorder=1)

color=iter(cmocean.cm.thermal(np.linspace(0.,0.9,len(mean))))
for i in range(len(mean)):
    ccc=next(color)
    plt.scatter(xx[i],mean[i],color=ccc,s=100,edgecolor=ccc,zorder=5)
    if i == 0:
        plt.errorbar(xx[i],mean[i],yerr=np.array([[mean[i]+p5[i],p95[i]-mean[i]]]),
                     color=ccc,linewidth=1.5,capthick=3,capsize=10,zorder=4)
    else:
        plt.errorbar(xx[i],mean[i],yerr=np.array([[mean[i]-p5[i],p95[i]-mean[i]]]).T,
                 color=ccc,linewidth=1.5,capthick=3,capsize=10,zorder=4)
    if i == 0:
        plt.text(xx[i],p95[i]-mean[i]-0.15,r'\textbf{%s}' % runnames[i],
                 color='k',fontsize=9,ha='center',va='center')
    else:
        plt.text(xx[i],p95[i]+0.15,r'\textbf{%s}' % runnames[i],
                 color='k',fontsize=9,ha='center',va='center')

if varnames[0] == 'T2M':       
    plt.ylabel(r'\textbf{%s ($^{\circ}$C decade$^{-1}$)}' % varnames[0],
                         color='k',fontsize=10)
plt.yticks(np.arange(-3,3.1,0.5),list(map(str,np.arange(-3,3.1,0.5))),fontsize=6)
plt.ylim([-2,1.5])

plt.savefig(directoryfigure + '%slTrends_%s_%s-%s.png' % (region,varnames[0],
            yearmn,yearmx),dpi=300)
    