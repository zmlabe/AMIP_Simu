"""
Script plots the "ration of predictable components" (RPC) as defined in
Eade et al. 2014 [GRL]. We compare the WACCM experiments with ERA-Interim.

Notes
-----
    Author : Zachary Labe
    Date   : 10 July 2019
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
directoryfigure = '/home/zlabe/Desktop/RPC/'

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
varnames = ['SLP','Z500','U200','U10','T2M','Z50','T850']
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
for v in range(len(varnames)):
    moddt,eradt,lat,lon = readVar(varnames[v])
    
    ### Calculate variances
    ensmean = np.nanmean(moddt,axis=1)
    ensmeanvari = np.var(ensmean,axis=1)
    ensvari = np.nanmean(np.var(moddt,axis=2),axis=1)
    
    ### Calculate pearson correlations
    corrs = np.empty((moddt.shape[0],moddt.shape[3],moddt.shape[4]))
    pvals = np.empty((moddt.shape[0],moddt.shape[3],moddt.shape[4]))
    for vv in range(moddt.shape[0]):
            for i in range(moddt.shape[3]):
                for j in range(moddt.shape[4]):        
                    xx = eradt[:,i,j]
                    yy = ensmean[vv,:,i,j]
                    na = np.logical_or(np.isnan(xx),np.isnan(yy))
                    corrs[vv,i,j],pvals[vv,i,j]=sts.pearsonr(xx[~na],yy[~na])
                    
    ### Calculate RPC
    rpc = corrs/(np.sqrt(ensmeanvari/ensvari))
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Plot correlations
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
    ### Set limits for contours and colorbars
    limit = np.arange(0,1.6,0.25)
    barlim = np.round(np.arange(0,1.6,0.5),2)
    fig = plt.figure()
    for i in range(rpc.shape[0]):
        var = rpc[i]
    #        pvar = pvalue[i]
        
        ax1 = plt.subplot(2,3,i+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
    #        pvar,lons_cyclic = addcyclic(pvar, lon)
    #        pvar,lons_cyclic = shiftgrid(180.,pvar,lons_cyclic,start=False)
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
                  
        circle = m.drawmapboundary(fill_color='white',color='dimgrey',
                                   linewidth=0.7)
        circle.set_clip_on(False)
        
        cs = m.contourf(x,y,var,limit,extend='max')
    #        cs1 = m.contourf(x,y,pvar,colors='None',hatches=['....'],
    #                     linewidths=0.4)
                  
        m.drawcoastlines(color='dimgray',linewidth=0.7)
        
        if varnames[v] == 'T2M':
            m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',
                             lakes=True,resolution='c',zorder=5)
                
        cmap = cmocean.cm.rain
        cs.set_cmap(cmap)   
        ax1.annotate(r'\textbf{%s}' % runnamesm[i],xy=(0,0),
                     xytext=(0.865,0.90),textcoords='axes fraction',
                     color='k',fontsize=11,rotation=320,ha='center',
                     va='center')

            
    ###########################################################################
    cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=True)
    
    cbar.set_label(r'\textbf{RPC}',fontsize=11,color='k',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim))) 
    cbar.ax.tick_params(axis='x', size=.01)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.dividers.set_color('dimgrey')
    cbar.dividers.set_linewidth(1.2)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=10)
    
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(bottom=0.16)
    
    plt.savefig(directoryfigure + 'RPC_%s.png' % (varnames[v]),
                dpi=300)