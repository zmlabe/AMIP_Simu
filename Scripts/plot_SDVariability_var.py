"""
Script plots standard deviation of various climate variables in reanalysis
and the WACCM experiments.

Notes
-----
    Author : Zachary Labe
    Date   : 22 February 2019
"""

### Import modules
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cmocean
#import os
#os.environ['PROJ_LIB'] = '/home/zlabe/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import read_MonthlyData as MOM
import read_Reanalysis as MOR
import calc_Utilities as UT
import scipy.stats as sts
import palettable.cubehelix as cm

### Define directories
directoryfigure = '/home/zlabe/Desktop/STD_DJF/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting Variability - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
varnames = ['U200']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readVar(varnames):
    ### Call function to read in ERA-Interim (detrended)
    lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',True,True)
    
    ### Call functions to read in WACCM data (detrended)
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                   'surface',True,True)
        
    ### Retrieve time period of interest
    modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                       era.shape[3]))
    for i in range(len(runnamesm)):
        for j in range(ensembles):
            modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                                lat,lon,'surface',1)
    eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    
    return modq,eraq,lat,lon

### Read in data
moddt,eradt,lat,lon = readVar(varnames[0])

### Calculate standard deviation for ERA-I
erastd = np.empty((eradt.shape[1],eradt.shape[2]))
for i in range(eradt.shape[1]):
    for j in range(eradt.shape[2]):
        erastd[i,j] = np.nanstd(eradt[:,i,j])
        
### Calculate standard deviation for models
modelstd = np.empty((moddt.shape[0],moddt.shape[1],moddt.shape[3],
                     moddt.shape[4]))
for mo in range(moddt.shape[0]):
    print('Completed: Standard deviation --> %s!' % runnamesm[mo])
    for ens in range(moddt.shape[1]):
        for i in range(moddt.shape[3]):
            for j in range(moddt.shape[4]):
                modelstd[mo,ens,i,j] = np.nanstd(moddt[mo,ens,:,i,j])

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for standard deviation
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

for v in range(len(varnames)):
    ### Set limits for contours and colorbars
    if varnames[v] == 'T2M':
        limit = np.arange(0,4.01,0.25)
        barlim = np.arange(0,5,2)
        cmap = cm.classic_16.mpl_colormap  
        label = r'\textbf{$^{\circ}$C}'
    elif varnames[v] == 'U200':
        limit = np.arange(0,10.1,0.25)
        barlim = np.arange(0,11,5)
        cmap = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
    elif varnames[v] == 'U700':
        limit = np.arange(0,5.1,0.25)
        barlim = np.arange(0,6,5)
        cmap = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
    elif varnames[v] == 'U10':
        limit = np.arange(0,20.1,0.25)
        barlim = np.arange(0,21,10)
        cmap = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
    elif varnames[v] == 'SLP':
        limit = np.arange(0,10.1,0.25)
        barlim = np.arange(0,11,5)
        cmap = cm.classic_16.mpl_colormap 
        label = r'\textbf{hPa}'
    elif varnames[v] == 'Z500':
        limit = np.arange(0,75.1,1)
        barlim = np.arange(-0,76,25)
        cmap = cm.classic_16.mpl_colormap 
        label = r'\textbf{m}'
        
    fig = plt.figure(figsize=(6,5))
    for i in range(len(runnames)):
        if i == 0:
            var = erastd
        else:
            var = np.nanmean(modelstd[i-1],axis=0)
        
        ax1 = plt.subplot(3,4,su[i]+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
           
        if i == 0:
            circle = m.drawmapboundary(fill_color='white',color='k',
                              linewidth=3)
            circle.set_clip_on(False)
        else:
            circle = m.drawmapboundary(fill_color='white',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
        
        cs = m.contourf(x,y,var,limit,extend='max')
        
        if varnames[v] == 'T2M':
            m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',
                         lakes=True,resolution='c',zorder=5)
            m.drawcoastlines(color='k',linewidth=0.7)
        else:
            m.drawcoastlines(color='k',linewidth=0.7)
                
        cs.set_cmap(cmap) 
        ax1.annotate(r'\textbf{%s}' % runnames[i],xy=(0,0),xytext=(0.865,0.91),
                     textcoords='axes fraction',color='k',fontsize=11,
                     rotation=320,ha='center',va='center')
    
    ###########################################################################
    cbar_ax = fig.add_axes([0.412,0.23,0.4,0.03])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0,hspace=0.01)
    
    plt.savefig(directoryfigure + 'SDVariability_%s_.png' % (varnames[v]),
                dpi=300)
    
###########################################################################
###########################################################################
###########################################################################
### Plot variable data for standard deviation differences from ERA-interim
for v in range(len(varnames)):
    ### Set limits for contours and colorbars
    if varnames[v] == 'T2M':
        limite = np.arange(0,4.01,0.25)
        barlime = np.arange(0,5,2)
        cmape = cm.classic_16.mpl_colormap 
        limit = np.arange(-2,2.1,0.1)
        barlim = np.arange(-2,3,2)
        cmap = cmocean.cm.balance 
        label = r'\textbf{$^{\circ}$C}'
    elif varnames[v] == 'U200':
        limite = np.arange(0,10.1,0.25)
        barlime = np.arange(0,11,5)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
        limit = np.arange(-4,4.1,0.25)
        barlim = np.arange(-4,5,4)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'U700':
        limite = np.arange(0,5.1,0.25)
        barlime = np.arange(0,6,5)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
        limit = np.arange(-4,4.1,0.25)
        barlim = np.arange(-4,5,4)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'U10':
        limite = np.arange(0,20.1,0.25)
        barlime = np.arange(0,21,10)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
        limit = np.arange(-4,4.1,0.25)
        barlim = np.arange(-4,5,4)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'SLP':
        limite = np.arange(0,10.1,0.25)
        barlime = np.arange(0,11,5)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{hPa}'
        limit = np.arange(-4,4.1,0.25)
        barlim = np.arange(-4,5,4)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'Z500':
        limite = np.arange(0,75.1,0.25)
        barlime = np.arange(0,76,25)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m}'
        limit = np.arange(-25,26.1,1)
        barlim = np.arange(-25,26,25)
        cmap = cmocean.cm.balance 
        
    fig = plt.figure(figsize=(6,5))
    for i in range(len(runnames)):
        if i == 0:
            var = erastd
        else:
            var = np.nanmean(modelstd[i-1],axis=0) - erastd
        
        ax1 = plt.subplot(3,4,su[i]+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
           
        if i == 0:
            circle = m.drawmapboundary(fill_color='white',color='k',
                              linewidth=3)
            circle.set_clip_on(False)
        else:
            circle = m.drawmapboundary(fill_color='white',color='dimgray',
                              linewidth=0.7)
            circle.set_clip_on(False)
        
        if i == 0:
            cse = m.contourf(x,y,var,limite,extend='max')
            m.drawcoastlines(color='k',linewidth=0.7)
        else:
            cs = m.contourf(x,y,var,limit,extend='both')
            m.drawcoastlines(color='dimgrey',linewidth=0.7)
        
        if varnames[v] == 'T2M':
            m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',
                         lakes=True,resolution='c',zorder=5)
                
        cse.set_cmap(cmape) 
        cs.set_cmap(cmap) 
        ax1.annotate(r'\textbf{%s}' % runnames[i],xy=(0,0),xytext=(0.865,0.91),
                     textcoords='axes fraction',color='k',fontsize=11,
                     rotation=320,ha='center',va='center')
    
    ###########################################################################
    cbar_ax = fig.add_axes([0.412,0.23,0.4,0.03])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
    
    ###########################################################################
    cbar_ax = fig.add_axes([0.055,0.51,0.2,0.02])                
    cbar = fig.colorbar(cse,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  
    
    cbar.set_ticks(barlime)
    cbar.set_ticklabels(list(map(str,barlime)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0,hspace=0.01)
    
    plt.savefig(directoryfigure + 'SDVariability_diff_%s_.png' % (varnames[v]),
                dpi=300)