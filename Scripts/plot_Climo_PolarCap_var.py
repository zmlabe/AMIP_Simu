"""
Script plots monthly climatology of various climate variables in reanalysis
and the WACCM experiments. These variables are averages over the polar cap.

Notes
-----
    Author : Zachary Labe
    Date   : 25 February 2019
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
directoryfigure = '/home/zlabe/Desktop/Climo_Monthly/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting Climatologies Polar Cap - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
varnames = ['TEMP','U','GEOP']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readVar(varnames,runnamesm):
    ### Call function to read in ERA-Interim
    lat,lon,lev,era = MOR.readDataRMeans(varnames)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2]))
    for i in range(len(runnamesm)):
        lat,lon,lev,models[i] = MOM.readDataMMeans(varnames,runnamesm[i])
        
    ### Slice months for January-February
    eraq = np.nanmean(era[:,0:2,:],axis=1)
    
    ### Slice months for January-February
    modelsq = np.nanmean(models[:,:,:,0:2,:],axis=3)
    
    return eraq,modelsq,lat,lon,lev

for v in range(len(varnames)):
    ### Read in data functions
    era,mod,lat,lon,lev = readVar(varnames[v],runnamesm)
    
    ### Calculate ensemble mean
    modm = np.nanmean(mod,axis=1)
    
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
    if varnames[v] == 'TEMP':
        limit = np.arange(-70,-9,1)
        barlim = np.arange(-70,-9,60)
        cmap = cmape = cm.classic_16.mpl_colormap 
        limitd = np.arange(-5,5.1,0.1)
        barlimd = np.arange(-5,6,5)
        cmapd = cmocean.cm.balance  
        time = np.arange(era.shape[0])
        label = r'\textbf{$^{\circ}$C}'
        zscale = np.array([1000,700,500,300])
    elif varnames[v] == 'U':
        limit = np.arange(-10,11,1)
        barlim = np.arange(-10,11,10)
        cmap = cmape = cm.classic_16.mpl_colormap 
        limitd = np.arange(-15,15.1,0.1)
        barlimd = np.arange(-15,16,15)
        cmapd = cmocean.cm.balance  
        time = np.arange(era.shape[0])
        label = r'\textbf{m/s}'
        zscale = np.array([1000,700,500,300,200,100,50,30,10])
    elif varnames[v] == 'GEOP':
        limit = np.arange(2000,3001,100)
        barlim = np.arange(2000,3001,1000)
        cmap = cmape = cm.classic_16.mpl_colormap 
        limitd = np.arange(-100,100.1,2)
        barlimd = np.arange(-100,101,50)
        cmapd = cmocean.cm.balance  
        time = np.arange(era.shape[0])
        label = r'\textbf{m}'
        zscale = np.array([1000,700,500,300])
    timeq,levq = np.meshgrid(time,lev)
            
    fig = plt.figure(figsize=(8,5))
    for i in range(modm.shape[0]+1):
        if i == 0:
            var = era
        else:
            var = np.fliplr(modm[i-1,:,:]) - era
            
        var = np.fliplr(var).transpose()
        
        ### Create plot
        ax1 = plt.subplot(3,4,su[i]+1)
        ax1.spines['top'].set_color('dimgrey')
        ax1.spines['right'].set_color('dimgrey')
        ax1.spines['bottom'].set_color('dimgrey')
        ax1.spines['left'].set_color('dimgrey')
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['right'].set_linewidth(2)
        ax1.spines['top'].set_linewidth(2)
        if i == 0 or i == 4:
            ax1.tick_params(axis='y',direction='out',which='major',pad=3,
                        width=2,color='dimgrey')
            plt.gca().axes.get_yaxis().set_visible(True)
        else:
            ax1.tick_params(axis='y',direction='out',which='major',pad=3,
                width=0,color='w')
            plt.gca().axes.get_yaxis().set_visible(False)
            
        if i == 0 or i == 4 or i ==5 or i ==6:
            ax1.tick_params(axis='x',direction='out',which='major',pad=3,
                        width=2,color='dimgrey')   
            plt.gca().axes.get_xaxis().set_visible(True)
        else:
            ax1.tick_params(axis='x',direction='out',which='major',pad=3,
                        width=0,color='w')  
            plt.gca().axes.get_xaxis().set_visible(False)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')
        
        if i == 0:
            cs = plt.contourf(timeq,levq,var,limit,extend='both')
            cs.set_cmap(cmap) 
        else:
            csd = plt.contourf(timeq,levq,var,limitd,extend='both')
            csd.set_cmap(cmapd)
        
        plt.gca().invert_yaxis()
        plt.yscale('log',nonposy='clip')
        
        plt.xlim([0,37])
        if varnames[v] == 'TEMP' or varnames[v] == 'GEOP':
            plt.ylim([1000,300])
        else:
            plt.ylim([1000,10])
        plt.xticks(np.arange(0,era.shape[0],10),map(str,
                   np.arange(1979,2017,10)),fontsize=6)
        plt.yticks(zscale,map(str,zscale),ha='right',fontsize=6)
        plt.minorticks_off()
               
        ax1.annotate(r'\textbf{%s}' % runnames[i],xy=(0,1000),xytext=(0.17,0.87),
             textcoords='axes fraction',color='k',fontsize=11,
             rotation=0,ha='center',va='center')
        
    ###########################################################################
    cbar_ax = fig.add_axes([0.412,0.28,0.4,0.03])                
    cbar = fig.colorbar(csd,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  
    
    cbar.set_ticks(barlimd)
    cbar.set_ticklabels(list(map(str,barlimd)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
    
    ###########################################################################
    cbar_ax = fig.add_axes([0.16,0.54,0.1,0.02])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
        
    plt.subplots_adjust(top=0.85,hspace=0.05,wspace=0.1)    
    plt.savefig(directoryfigure + 'Vertical_PolarCap_%s.png' % varnames[v],
                dpi=300)
    print('Completed: Script done!')