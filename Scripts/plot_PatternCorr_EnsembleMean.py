"""
Script calculates and plots pattern correlations for various variables.
Correlations are computed between ERA-Interim and the model experiments.

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

### Define directories
directoryfigure = '/home/zlabe/Desktop/PatternCorr/Detrended/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting Pattern Correlations - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
period = 'N'
varnames = ['SLP','Z500','U200','U10','T2M',]
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
    elif period == 'DJF':
        modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                           era.shape[3]))
        for i in range(len(runnamesm)):
            for j in range(ensembles):
                modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:,:],
                                                    lat,lon,'surface',1)
        eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    
    return modq,eraq,lat,lon

### Read data from functions
for rr in range(len(varnames)):
    mod,era,lat,lon = readVar(varnames[rr],runnamesm,period)
    
#    eanom = era - np.nanmean(era,axis=0)
#    modanom = np.empty((mod.shape[0],mod.shape[1],mod.shape[2],mod.shape[3],
#                        mod.shape[4]))
#    for ru in range(modanom.shape[0]):
#        for ens in range(modanom.shape[1]):
#            modanom[ru,ens,:,:,:] = mod[ru,ens,:,:,:] - np.nanmean(
#                                            mod[ru,ens,:,:,:],axis=0)
#    
    #### Calculate correlation coefficients at each grid point
    #corrs = np.empty((mod.shape[0],mod.shape[1],mod.shape[3],mod.shape[4]))
    #for ru in range(mod.shape[0]):
    #    for ens in range(mod.shape[1]):
    #        for i in range(mod.shape[3]):
    #            for j in range(mod.shape[4]):
    #                corrs[ru,ens,i,j] = sts.pearsonr(era[:,i,j],
    #                                                 mod[ru,ens,:,i,j])[0]
    #                
    #### Calculate ensemble mean
    #corrm = np.nanmean(corrs,axis=1)      
    modq = np.nanmean(mod,axis=1)
        
    ### Calculate correlation coefficients at each grid point
    corrm = np.empty((mod.shape[0],mod.shape[3],mod.shape[4]))
    pvalue = np.empty((mod.shape[0],mod.shape[3],mod.shape[4]))
    for ru in range(mod.shape[0]):
        for i in range(mod.shape[3]):
            for j in range(mod.shape[4]):
                xx = era[:,i,j]
                yy = modq[ru,:,i,j]
                na = np.logical_or(np.isnan(xx),np.isnan(yy))
                corrm[ru,i,j],pvalue[ru,i,j] = sts.pearsonr(xx[~na],yy[~na])
                
    ### Significant at 95% confidence level
    pvalue[np.where(pvalue >= 0.05)] = np.nan
    pvalue[np.where(pvalue < 0.05)] = 1.
                    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Plot correlations
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
    ### Set limits for contours and colorbars
    limit = np.arange(-0.6,0.61,0.1)
    barlim = np.around(np.arange(-0.6,0.7,0.2),1)
    fig = plt.figure()
    for i in range(len(runnamesm)):
        var = corrm[i]
        pvar = pvalue[i]
        
        ax1 = plt.subplot(2,3,i+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
        pvar,lons_cyclic = addcyclic(pvar, lon)
        pvar,lons_cyclic = shiftgrid(180.,pvar,lons_cyclic,start=False)
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
                  
        circle = m.drawmapboundary(fill_color='white',color='dimgrey',
                                   linewidth=0.7)
        circle.set_clip_on(False)
        
        cs = m.contourf(x,y,var,limit,extend='both')
        cs1 = m.contourf(x,y,pvar,colors='None',hatches=['....'],
                     linewidths=0.4)
                  
        m.drawcoastlines(color='dimgray',linewidth=0.7)
        
        if varnames[rr] == 'T2M':
            m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',
                             lakes=True,resolution='c',zorder=5)
                
        cs.set_cmap(cmocean.cm.balance) 
        ax1.annotate(r'\textbf{%s}' % runnamesm[i],xy=(0,0),
                     xytext=(0.865,0.90),textcoords='axes fraction',
                     color='k',fontsize=11,rotation=320,ha='center',
                     va='center')
    
                
    ###########################################################################
    cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(r'\textbf{r}',fontsize=11,color='k',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
    
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(bottom=0.16)
    
    plt.savefig(directoryfigure + \
                '%s/Correlation_Mean_%s_NOTrend_%s.png' % (period,
                                                           varnames[rr],
                                                           period),dpi=300)