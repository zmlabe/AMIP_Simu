"""
Script plots trends of various forcings for multiple time periods in each
model simulation.

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

### Define directories
directoryfigure = '/home/zlabe/Desktop/Trends/'

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting WACC Forcing Field Trends - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
periodq = ['AMJ','Annual','D','DJF','FM','JF','JJA','N','O','ON','OND','S']
yearmnq = [1980,1991,1995,2000,2005]
yearmx = 2015
varnames = ['LHFLX','SHFLX']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

for pp in range(len(periodq)):
    period = periodq[pp]
    for i in range(len(yearmnq)):
        yearmn = yearmnq[i]
        for v in range(len(varnames)):  
            ### Call function to read in ERA-Interim
            lat,lon,time,lev,era = MOR.readDataR('T2M','surface',False,True)
            
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
            elif period == 'JF':
                modq = np.nanmean(models[:,:,:,0:2,:,:],axis=3)
            elif period == 'S':
                modq = models[:,:,:,-4,:,:].squeeze()
            elif period == 'O':
                modq = models[:,:,:,-3,:,:].squeeze()
            elif period == 'N':
                modq = models[:,:,:,-2,:,:].squeeze()
            elif period == 'D':
                modq = models[:,:,:,-1:,:,:].squeeze()
            elif period == 'ON':
                modq = np.nanmean(models[:,:,:,9:11,:,:],axis=3)
            elif period == 'OND':
                modq = np.nanmean(models[:,:,:,-3:,:,:],axis=3)
            elif period == 'ND':
                modq = np.nanmean(models[:,:,:,-2:,:,:],axis=3)
            elif period == 'FM':
                modq = np.nanmean(models[:,:,:,1:3,:,:],axis=3) 
            elif period == 'JJA':
                modq = np.nanmean(models[:,:,:,5:8,:,:],axis=3)
            elif period == 'AMJ':
                modq = np.nanmean(models[:,:,:,3:6,:,:],axis=3)
            elif period == 'Annual':
                modq = np.nanmean(models[:,:,:,:,:,:],axis=3)        
                
            ### Calculate the trend for WACCM
            sliceq = np.where((years >= yearmn) & (years <= yearmx))[0]    
            
            modtrend = np.empty((len(runnamesm),ensembles,models.shape[4],
                                 models.shape[5]))
            for i in range(len(runnamesm)):
                modtrend[i,:,:,:] = UT.detrendData(modq[i],years,'surface',
                                                    yearmn,yearmx)
                print('Completed: Simulation --> %s!' % runnamesm[i])
                
            ### Calculate decadal trend
            dectrend = modtrend * 10.
            
            ### Take ensemble mean        
            modqq = np.nanmean(modq,axis=1)
            
            ### Mann-Kendall Trend test for each model and grid point
            pmodel = np.empty((modqq.shape[0],modqq.shape[2],modqq.shape[3]))
            for r in range(modqq.shape[0]):
                print('Completed: Simulation MK Test --> %s!' % runnamesm[r])
                for i in range(modqq.shape[2]):
                    for j in range(modqq.shape[3]):
                        trend,h,pmodel[r,i,j],z = UT.mk_test(modqq[r,sliceq,i,j],0.05)
            
            ###########################################################################
            ###########################################################################
            ###########################################################################
            ### Plot variable data for trends
            plt.rc('text',usetex=True)
            plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
            
            ### Set limits for contours and colorbars
            if varnames[v] == 'SST':
                limit = np.arange(-2,2.01,0.05)
                barlim = np.arange(-2,3,1)
                cmap = cmocean.cm.balance
                label = r'\textbf{$^{\circ}$C decade$^{-1}$}'
            elif varnames[v] == 'SIC':
                limit = np.arange(-20,20.1,0.5)
                barlim = np.arange(-20,21,10)
                cmap = cmocean.cm.balance_r
                label = r'\textbf{\% decade$^{-1}$}'
            elif varnames[v] == 'SNC':
                limit = np.arange(-10,10.1,0.5)
                barlim = np.arange(-10,11,10)
                cmap = cmocean.cm.balance_r
                label = r'\textbf{\% decade$^{-1}$}'
            elif varnames[v] in ('RNET','LHFLX','SHFLX'):
                limit = np.arange(-25,25.1,0.25)
                barlim = np.arange(-25,26,25)
                cmap = cmocean.cm.balance
                label = r'\textbf{W/m$^{2}$ decade$^{-1}$}'
                
            fig = plt.figure()
            for i in range(len(runnamesm)):
                if varnames[v] in ('RNET','LHFLX','SHFLX'):
                    var = np.nanmean(dectrend[i],axis=0)*-1. # upward flux == pos.
                else:
                    var = np.nanmean(dectrend[i],axis=0)
                pvar = pmodel[i]
                
                ax1 = plt.subplot(2,3,i+1)
                if varnames[v] in ('SST','SNC','RNET','LHFLX','SHFLX'):
                    m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                                area_thresh=10000.)
                elif varnames[v] == 'SIC':
                    m = Basemap(projection='npstere',boundinglat=49,lon_0=0,
                                resolution='l',round =True,area_thresh=10000.)
                
                var, lons_cyclic = addcyclic(var, lon)
                var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
                pvar,lons_cyclic = addcyclic(pvar, lon)
                pvar,lons_cyclic = shiftgrid(180.,pvar,lons_cyclic,start=False)
                lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
                x, y = m(lon2d, lat2d)
                   
                circle = m.drawmapboundary(fill_color='white',color='dimgray',
                                  linewidth=0.7)
                circle.set_clip_on(False)
                
                cs = m.contourf(x,y,var,limit,extend='both')
                cs1 = m.contourf(x,y,pvar,colors='None',hatches=['....'],
                                 linewidths=0.4)
                          
                if varnames[v] in ('SST','SIC','RNET','LHFLX','SHFLX'):               
                    m.drawcoastlines(color='darkgray',linewidth=0.3)
                    m.fillcontinents(color='dimgrey')
                elif varnames[v] == 'SNC':
                    m.drawcoastlines(color='darkgray',linewidth=0.3)
                    m.drawlsmask(land_color=(0,0,0,0),ocean_color='dimgrey',
                                 lakes=True,resolution='c',zorder=5)
                        
                cs.set_cmap(cmap) 
                ax1.annotate(r'\textbf{%s}' % runnamesm[i],xy=(0,0),xytext=(0.865,0.91),
                             textcoords='axes fraction',color='k',fontsize=11,
                             rotation=320,ha='center',va='center')
            
            ###########################################################################
            cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                 
            cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                                extend='both',extendfrac=0.07,drawedges=False)
            
            cbar.set_label(label,fontsize=11,color='dimgrey',labelpad=1.4)  
            
            cbar.set_ticks(barlim)
            cbar.set_ticklabels(list(map(str,barlim)))
            cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
            cbar.outline.set_edgecolor('dimgrey')
            
        
            plt.subplots_adjust(wspace=0.01,hspace=0.01,bottom=0.15)
        #    plt.subplots_adjust(top=0.85,wspace=0,hspace=0.01)
            
            plt.savefig(directoryfigure + 'SST_SeaIce/%s/Forcing_Trend_Models_%s_%s-%s.png' % (period,
                                                                                             varnames[v],
                                                                                             yearmn,yearmx),dpi=300)
            