"""
Script calculates trends for temperature profiles over the polar cap. We
assess warming contributions from SST and sea ice compared to reanalysis.

Notes
-----
    Author : Zachary Labe
    Date   : 17 July 2019
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
import palettable.cubehelix as cm

### Define directories
directoryfigure = '/home/zlabe/Desktop/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Calculating Polar Cap Warming - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2015
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
varnames = 'TEMP'
su = [0,1,2,4,5]
runnames = [r'ERAi',r'AMIP--ERAi',r'SST+SIC',r'[SST+SIC]--ERAi',r'[SST+SIC]--AMIP']
runnamesm = [r'CSST',r'CSIC',r'AMIP']
monthstext = [r'OCT',r'NOV',r'DEC',r'JAN',r'FEB',r'MAR']

def readVar(varnames,runnamesm):
    ### Call function to read in ERA-Interim
    lat,lon,lev,era = MOR.readDataRMeans(varnames)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2]))
    for i in range(len(runnamesm)):
        lat,lon,lev,models[i] = MOM.readDataMMeans(varnames,runnamesm[i])
            
    ### Arrange ERAI monthly time series
    eraravel = np.reshape(era.copy(),
                       (int(era.shape[0]*12),lev.shape[0]))
    eramonth = np.empty((era.shape[0]-1,6,lev.shape[0]))
    eramonth = []
    for i in range(9,eraravel.shape[0]-12,12):
        eramonthq = eraravel[i:i+6,:]
        eramonth.append(eramonthq)
    eramonth = np.asarray(eramonth)
    
    ### Arrange modeled monthly time series
    modravel = np.reshape(models.copy(),
                       (models.shape[0],models.shape[1],
                        int(models.shape[2]*12),lev.shape[0]))
    
    modelmonth = []
    for rr in range(models.shape[0]):
        for ens in range(models.shape[1]):
            modmonth = []
            for i in range(9,modravel.shape[2]-12,12):
                modmonthq = modravel[rr,ens,i:i+6,:]
                modmonth.append(modmonthq)
            modelmonth.append(modmonth)
    modelmonth = np.reshape(modelmonth,(models.shape[0],models.shape[1],
                                        models.shape[2]-1,6,
                                        lev.shape[0]))
            
    return eramonth,modelmonth,lat,lon,lev

###############################################################################
### Read in data functions
era,mod,lat,lon,lev = readVar(varnames,runnamesm)

### Calculate ensemble mean
modm = np.nanmean(mod,axis=1)

erar = UT.detrendDataR(era,years,'surface',year1,year2)[0]
modr = UT.detrendData(modm,years,'surface',year1,year2)

### Calculate decadal trends
eradr = erar * 10.
moddr = modr * 10. 
###############################################################################
### Mann-Kendall Trend test for ERAi and grid point
pera = np.empty((erar.shape))
for i in range(erar.shape[0]):
    for j in range(erar.shape[1]):
        trend,h,pera[i,j],z = UT.mk_test(era[:,i,j],0.05)
        
### Mann-Kendall Trend test for each model and grid point
pmodel = np.empty((modm.shape[0],modm.shape[2],modm.shape[3]))
for r in range(modm.shape[0]):
    print('Completed: Simulation MK Test --> %s!' % runnamesm[r])
    for i in range(modm.shape[2]):
        for j in range(modm.shape[3]):
            trend,h,pmodel[r,i,j],z = UT.mk_test(modm[r,:,i,j],0.05)

###############################################################################            
### Compare warming contributions - model simulations [r'CSST',r'CSIC',r'AMIP']
sic = moddr[0,:,:].transpose()
sst = moddr[1,:,:].transpose()
com = moddr[2,:,:].transpose()
rer = np.fliplr(eradr[:,:]).transpose() # [6,16] or [months,level]   

add = sic + sst # combined warming of sst and sic experiments
dif = add - com # difference between AMIP experiment and combined sst/sic
red = com - rer # difference between AMIP experiment and reanalysis        
rea = add - rer # difference between combined sst/sic and reanalysis

plotall = [rer,red,add,rea,dif]
              
###########################################################################
###########################################################################
###########################################################################
##### Plot profiles
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

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
        
### Set limits for contours and colorbars
limit = np.arange(-1,1.001,0.1)
barlim = np.arange(-1,2,1)
cmap = cmocean.cm.balance
label = r'\textbf{$^{\circ}$C decade$^{-1}$}'
zscale = np.array([1000,700,500,300,200,100])
timeq,levq = np.meshgrid(np.arange(6),lev)
        
fig = plt.figure()
for i in range(len(plotall)):
    if i == 0:
        var = plotall[i]
    else:
        var = plotall[i]
        
    var = var
    peraq = np.fliplr(pera).transpose()
    
    ### Create plot
    ax1 = plt.subplot(3,3,su[i]+1)
    ax1.spines['top'].set_color('dimgrey')
    ax1.spines['right'].set_color('dimgrey')
    ax1.spines['bottom'].set_color('dimgrey')
    ax1.spines['left'].set_color('dimgrey')
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    if i == 0 or i == 3:
        ax1.tick_params(axis='y',direction='out',which='major',pad=3,
                    width=2,color='dimgrey')
        plt.gca().axes.get_yaxis().set_visible(True)
    else:
        ax1.tick_params(axis='y',direction='out',which='major',pad=3,
            width=0,color='w')
        plt.gca().axes.get_yaxis().set_visible(False)
        
    if i==0 or i==3 or i==4 or i==5:
        ax1.tick_params(axis='x',direction='out',which='major',pad=3,
                    width=2,color='dimgrey')   
        plt.gca().axes.get_xaxis().set_visible(True)
    else:
        ax1.tick_params(axis='x',direction='out',which='major',pad=3,
                    width=0,color='w')  
        plt.gca().axes.get_xaxis().set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    cs = plt.contourf(timeq,levq,var,limit,extend='both')
    cs.set_cmap(cmap) 
    
    if i == 0:
        cs1 = plt.contourf(timeq,levq,peraq,colors='None',
                           hatches=['//////'],linewidths=0.4)
#    if i == 3:
#        pmodelq = pmodel[i-1].transpose()
#        cs1 = plt.contourf(timeq,levq,pmodelq,colors='None',
#                           hatches=['//////'],linewidths=0.4)
    
    plt.gca().invert_yaxis()
    plt.yscale('log',nonposy='clip')
    
    plt.xlim([0,5])
    plt.ylim([1000,100])
    plt.xticks(np.arange(0,6,1),monthstext,fontsize=6)
    plt.yticks(zscale,map(str,zscale),ha='right',fontsize=6)
    plt.minorticks_off()
           
    ax1.annotate(r'\textbf{%s}' % runnames[i],xy=(0,1000),xytext=(0.5,1.1),
         textcoords='axes fraction',color='k',fontsize=11,
         rotation=0,ha='center',va='center')
    
###########################################################################
cbar_ax = fig.add_axes([0.47,0.2,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar.outline.set_edgecolor('dimgrey')

plt.tight_layout()  
plt.subplots_adjust(top=0.85,wspace=0.14)  
plt.savefig(directoryfigure + 'Vertical_PolarCap_WarmingContr.png',dpi=300)
print('Completed: Script done!')