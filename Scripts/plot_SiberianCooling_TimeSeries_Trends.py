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
year1 = 1995
year2 = 2014
years = np.arange(1979,2016+1,1)

### Add parameters
ensembles = 10
monthperiod = ['DJF']
varnames = ['T2M']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
sliceq = True

def readVar(varnames,monthperiod):
    ### Call function to read in ERA-Interim
    lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',False,True)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                   'surface',False,True)
        
    ### Retrieve time period of interest
    if monthperiod == 'DJF':
        modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                           era.shape[3]))
        for i in range(len(runnamesm)):
            for j in range(ensembles):
                modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                                    lat,lon,'surface',1)
        eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    elif monthperiod == 'Annual':
        modq = np.nanmean(models[:,:,:,:,:,:],axis=3)
        eraq = np.nanmean(era[:,:,:,:],axis=1)
    
    ### Take ensemble mean
    modmean = np.nanmean(modq,axis=1)
    
    ### Slice over Arctic polar cap
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
eraanom,modelanom = readVar(varnames[0],monthperiod[0])

if sliceq == True:
    yearq = np.where((years >= year1) & (years <= year2))[0]
    eraanom = eraanom[yearq]
    modelanom = modelanom[:,yearq]
    
corrs = np.empty((modelanom.shape[0]))
for i in range(modelanom.shape[0]):
    corrs[i] = sts.pearsonr(eraanom,modelanom[i])[0]
    
### Calculate decadal trends for time series
def trend1D(yy):
    xx = np.arange(yy.shape[0])
    slopes,intercepts,r_value,p_value,std_err = sts.linregress(xx,yy)
    return slopes*10.,intercepts,r_value**2, p_value, std_err*10.

### Call trends
slopemod = np.empty((modelanom.shape[0]))
intermod = np.empty((modelanom.shape[0]))
rmod = np.empty((modelanom.shape[0]))
pmod = np.empty((modelanom.shape[0]))
std_errmod = np.empty((modelanom.shape[0]))
for i in range(modelanom.shape[0]):
    slopemod[i],intermod[i],rmod[i], \
                                pmod[i],std_errmod[i] = trend1D(modelanom[i,:])
sloper,interr,rr,pr,std_errr = trend1D(eraanom)

### Create single array for temperature trends
mean = np.append(sloper,slopemod)
std_err = np.append(std_errr, std_errmod)

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

plt.axhline(y=mean[0],linewidth=1,color='dimgrey',linestyle='--',
            dashes=(1,0.3),zorder=1)

color=iter(cmocean.cm.thermal(np.linspace(0.,0.9,len(mean))))
for i in range(len(mean)):
    ccc=next(color)
    plt.scatter(xx[i],mean[i],color=ccc,s=100,edgecolor=ccc,zorder=5,
                clip_on=False)
    plt.errorbar(xx[i],mean[i],yerr=std_err[i],
                 color=ccc,linewidth=1.5,capthick=3,capsize=7,zorder=4,
                 clip_on=False)

    plt.text(xx[i],std_err[i]+mean[i]+0.05,r'\textbf{%s}' % runnames[i],
             color='k',fontsize=9,ha='center',va='center',clip_on=False)
    
    if i > 0:
        plt.text(xx[i],mean[i]-std_err[i]-0.07,r'\textbf{%s}' % np.round(corrs[i-1],2),
             color='k',fontsize=9,ha='center',va='center',clip_on=False)

if varnames[0] == 'T2M':       
    plt.ylabel(r'\textbf{%s ($^{\circ}$C decade$^{-1}$)}' % varnames[0],
                         color='k',fontsize=10)
plt.yticks(np.arange(-3,3.1,0.25),list(map(str,np.arange(-3,3.1,0.25))),fontsize=6)
plt.ylim([-1.5,0.75])

plt.savefig(directoryfigure + 'Siberia_Trends_%s_%s-%s_%s.png' % (varnames[0],
            year1,year2,monthperiod[0]),dpi=300)