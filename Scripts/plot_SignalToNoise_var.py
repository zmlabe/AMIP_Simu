"""
Plot maps of AMIP data for each period using signal-to-noise ratios

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
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import read_MonthlyData as MOM
import read_Reanalysis as MOR
import calc_Utilities as UT

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting AMIP SNR - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
period = 'JFM'
varnames = ['T2M','SLP','Z500','Z50','U200','U10','THICK']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

### Define directories
directoryfigure = '/home/zlabe/Desktop/SNR/%s/' % period

for v in range(len(varnames)):
    ### Call function to read in ERA-Interim
    lat,lon,time,lev,era = MOR.readDataR(varnames[v],'surface',False,True)
    
    ### Call functions to read in WACCM data
    models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                       era.shape[2],era.shape[3]))
    for i in range(len(runnamesm)):
        lat,lon,time,lev,models[i] = MOM.readDataM(varnames[v],runnamesm[i],
                                                   'surface',False,True)
        
    ### Retrieve time period of interest
    if period == 'DJF':  
        modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                           era.shape[3]))
        for i in range(len(runnamesm)):
            for j in range(ensembles):
                modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                                    lat,lon,'surface',1)
        eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    elif period == 'JF':
        modq = np.nanmean(models[:,:,:,0:2,:,:],axis=3)
        eraq = np.nanmean(era[:,0:2,:,:],axis=1)
    elif period == 'JFM':
        modq = np.nanmean(models[:,:,:,0:3,:,:],axis=3)
        eraq = np.nanmean(era[:,0:3,:,:],axis=1)
    elif period == 'ON':
        modq = np.nanmean(models[:,:,:,9:11,:,:],axis=3)
        eraq = np.nanmean(era[:,9:11,:,:],axis=1)
    elif period == 'OND':
        modq = np.nanmean(models[:,:,:,-3:,:,:],axis=3)
        eraq = np.nanmean(era[:,-3:,:,:],axis=1)
    elif period == 'S':
        modq = models[:,:,:,-4,:,:].squeeze()
        eraq = era[:,-4,:,:].squeeze()
    elif period == 'O':
        modq = models[:,:,:,-3,:,:].squeeze()
        eraq = era[:,-3,:,:].squeeze()
    elif period == 'N':
        modq = models[:,:,:,-2,:,:].squeeze()
        eraq = era[:,-2,:,:].squeeze()
    elif period == 'D':
        modq = models[:,:,:,-1:,:,:].squeeze()
        eraq = era[:,-1:,:,:].squeeze()
    elif period == 'ND':
        modq = np.nanmean(models[:,:,:,-2:,:,:],axis=3)
        eraq = np.nanmean(era[:,-2:,:,:],axis=1)
    elif period == 'FM':
        modq = np.nanmean(models[:,:,:,1:3,:,:],axis=3)
        eraq = np.nanmean(era[:,1:3,:,:],axis=1)   
    elif period == 'JJA':
        modq = np.nanmean(models[:,:,:,5:8,:,:],axis=3)
        eraq = np.nanmean(era[:,5:8,:,:],axis=1) 
    elif period == 'AMJ':
        modq = np.nanmean(models[:,:,:,3:6,:,:],axis=3)
        eraq = np.nanmean(era[:,3:6,:,:],axis=1) 
    elif period == 'Annual':
        modq = np.nanmean(models[:,:,:,:,:,:],axis=3)
        eraq = np.nanmean(era[:,:,:,:],axis=1)           
        
    ### Calculate the trend for WACCM
    yearmn = 1980
    yearmx = 2016
    sliceq = np.where((years >= yearmn) & (years <= yearmx))[0]    
    
    ### Calculate ensemble mean
    ensmean = np.nanmean(modq,axis=1)
    
    modtrend = np.empty((len(runnamesm),ensembles,models.shape[4],
                         models.shape[5]))
    for i in range(len(runnamesm)):
        modtrend[i,:,:,:] = UT.detrendData(modq[i],years,'surface',
                                            yearmn,yearmx)
        print('Completed: Simulation --> %s!' % runnamesm[i])

    meantrend = UT.detrendData(ensmean,years,'surface',yearmn,yearmx)
        
    ### Calculate decadal trend
    dectrend = modtrend * 10.
    dectrendm = meantrend * 10.
    
    ### Calculate SNR
    ensstd = np.nanstd(dectrend,axis=1)
    
    snr = np.abs(dectrendm)/ensstd
    snr[np.where(snr <= 0.)] = 0.
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Plot variable data for trends
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
    ### Set limits for contours and colorbars
    limit = np.arange(0,5.1,0.25)
    barlim = np.round(np.arange(0,5.1,1),2)
        
    fig = plt.figure()
    for i in range(len(runnamesm)):
        var = snr[i]
        
        ax1 = plt.subplot(2,3,i+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='white',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs = m.contourf(x,y,var,limit,extend='max')
                  
        m.drawcoastlines(color='dimgray',linewidth=0.7)
                
        cs.set_cmap(cmocean.cm.matter_r) 
        ax1.annotate(r'\textbf{%s}' % runnamesm[i],xy=(0,0),xytext=(0.865,0.91),
                     textcoords='axes fraction',color='k',fontsize=11,
                     rotation=320,ha='center',va='center')
    
    ###########################################################################
    cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                 
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=True)
    
    cbar.set_label(r'\textbf{Signal-to-Noise Ratio}',
                   fontsize=11,color='dimgrey',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim))) 
    cbar.ax.tick_params(axis='x', size=.01)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.dividers.set_color('dimgrey')
    cbar.dividers.set_linewidth(1.2)
    cbar.outline.set_linewidth(1.2)
    cbar.ax.tick_params(labelsize=6)
    
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(bottom=0.16)
    
    plt.savefig(directoryfigure + 'SNR_%s_%s-%s.png' % (varnames[v],
                yearmn,yearmx),dpi=300)
    