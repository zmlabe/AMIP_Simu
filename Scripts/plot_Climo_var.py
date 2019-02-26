"""
Script plots DJF climatology of various climate variables in reanalysis
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
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import read_MonthlyData as MOM
import read_Reanalysis as MOR
import calc_Utilities as UT
import scipy.stats as sts
import palettable.cubehelix as cm

### Define directories
directoryfigure = '/home/zlabe/Desktop/Climo_DJF/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting Climatologies - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
varnames = ['T2M','SLP','Z500','U200','U10']
varnames = ['T2M']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

def readVar(varnames,runnamesm):
    ### Call function to read in ERA-Interim (detrended)
    lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',False,True)
    
    ### Call functions to read in WACCM data (detrended)
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
            modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:,:],
                                                lat,lon,'surface',1)
    eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    
    return modq,eraq,lat,lon

#### TEST
#import scipy.signal as ss
#b = np.empty(a.shape)
#for i in range(a.shape[1]):
#    for j in range(b.shape[2]):
#        b[:,i,j] = ss.detrend(a[:,i,j],type='linear')
#plt.contourf(b[0],np.arange(-30,31,1),cmap=cmocean.cm.balance,
#             extend='both')

###############################################################################
###############################################################################
###############################################################################
### Plot variable data for climatological values
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

for v in range(len(varnames)):
    ### Read in data
    moddt,eradt,lat,lon = readVar(varnames[v],runnamesm)
    
    ### Set limits for contours and colorbars
    if varnames[v] == 'T2M':
        limite = np.arange(-30,0.1,1)
        barlime = np.arange(-30,1,30)
        cmape = cm.classic_16.mpl_colormap 
        limit = np.arange(-10,10.1,1)
        barlim = np.arange(-10,11,5)
        cmap = cmocean.cm.balance 
        label = r'\textbf{$^{\circ}$C}'
    elif varnames[v] == 'U200':
        limite = np.arange(0,70.1,5)
        barlime = np.arange(0,71,35)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
        limit = np.arange(-8,8.1,1)
        barlim = np.arange(-8,9,4)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'U10':
        limite = np.arange(-30,81,5)
        barlime = np.arange(-30,81,110)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m/s}'
        limit = np.arange(-10,10.1,2)
        barlim = np.arange(-10,11,5)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'SLP':
        limite = np.arange(990,1041,2)
        barlime = np.arange(990,1041,50)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{hPa}'
        limit = np.arange(-10,10.1,1)
        barlim = np.arange(-10,11,5)
        cmap = cmocean.cm.balance 
    elif varnames[v] == 'Z500':
        limite = np.arange(5000,6001,100)
        barlime = np.arange(5000,6001,500)
        cmape = cm.classic_16.mpl_colormap 
        label = r'\textbf{m}'
        limit = np.arange(-50,50.1,5)
        barlim = np.arange(-50,51,50)
        cmap = cmocean.cm.balance 
        
    fig = plt.figure(figsize=(6,5))
    for i in range(len(runnames)):
        if i == 0:
            var = np.nanmean(eradt,axis=0)
        else:
            varq = np.nanmean(moddt[i-1],axis=0) - eradt
            var = np.nanmean(varq,axis=0)
        
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
            cse = m.contourf(x,y,var,limite,extend='both')
            m.drawcoastlines(color='k',linewidth=0.7)
        else:
            cs = m.contourf(x,y,var,limit,extend='both')
            m.drawcoastlines(color='dimgrey',linewidth=0.7)
        
#        if varnames[v] == 'T2M':
#            m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',
#                         lakes=True,resolution='c',zorder=5)
                
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
    
    plt.savefig(directoryfigure + 'Climo_diff_%s_.png' % (varnames[v]),
                dpi=300)