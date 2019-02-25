"""
Script plots trends of various variables over the WACC period. Subplot compares
all six experiments with ERA-Interim.

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
varnames = ['T2M','SLP','Z500','Z50','U200','U10']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

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
    modq = np.empty((len(runnamesm),ensembles,era.shape[0]-1,era.shape[2],
                       era.shape[3]))
    for i in range(len(runnamesm)):
        for j in range(ensembles):
            modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                                lat,lon,'surface',1)
    eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)
    
    ### Calculate the trend for WACCM
    yearmn = 2005
    yearmx = 2014
    sliceq = np.where((years >= yearmn) & (years <= yearmx))[0]    
    
    modtrend = np.empty((len(runnamesm),ensembles,models.shape[4],
                         models.shape[5]))
    for i in range(len(runnamesm)):
        modtrend[i,:,:,:] = UT.detrendData(modq[i],years,'surface',
                                            yearmn,yearmx)
        print('Completed: Simulation --> %s!' % runnamesm[i])
        
    ### Calculate decadal trend
    dectrend = modtrend * 10.
        
    ### Calculate the trend for ERA-Interim
    retrend = UT.detrendDataR(eraq,years,'surface',yearmn,yearmx)
    
    #### Calculate decadal trend
    redectrend = retrend * 10.
    
    ### Mann-Kendall Trend test for ERAi and grid point
    pera = np.empty((redectrend.shape))
    for i in range(redectrend.shape[0]):
        for j in range(redectrend.shape[1]):
            trend,h,pera[i,j],z = UT.mk_test(eraq[sliceq,i,j],0.05)
    
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
    if varnames[v] == 'T2M':
        limit = np.arange(-2,2.01,0.05)
        barlim = np.arange(-2,3,1)
        cmap = cmocean.cm.balance
        label = r'\textbf{$^{\circ}$C decade$^{-1}$}'
    elif varnames[v] == 'Z500':
        limit = np.arange(-20,20.1,1)
        barlim = np.arange(-20,21,10)
        cmap = cmocean.cm.balance
        label = r'\textbf{m decade$^{-1}$}'
    elif varnames[v] == 'Z50':
        limit = np.arange(-50,50.1,1)
        barlim = np.arange(-50,51,25)
        cmap = cmocean.cm.balance
        label = r'\textbf{m decade$^{-1}$}'
    elif varnames[v] == 'U200':
        limit = np.arange(-5,5.1,0.25)
        barlim = np.arange(-5,6,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/s decade$^{-1}$}'
    elif varnames[v] == 'U10':
        limit = np.arange(-5,5.1,0.25)
        barlim = np.arange(-5,6,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/s decade$^{-1}$}'
    elif varnames[v] == 'SLP':
        limit = np.arange(-3,3.1,0.25)
        barlim = np.arange(-3,4,3)
        cmap = cmocean.cm.balance
        label = r'\textbf{hPa decade$^{-1}$}'
        
    fig = plt.figure(figsize=(6,5))
    for i in range(len(runnames)):
        if i == 0:
            var = redectrend
            pvar = pera
        else:
            var = np.nanmean(dectrend[i-1],axis=0)
            pvar = pmodel[i-1]
        
        ax1 = plt.subplot(3,4,su[i]+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        pvar,lons_cyclic = addcyclic(pvar, lon)
        pvar,lons_cyclic = shiftgrid(180.,pvar,lons_cyclic,start=False)
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
        
        cs = m.contourf(x,y,var,limit,extend='both')
        cs1 = m.contourf(x,y,pvar,colors='None',hatches=['....'],
                         linewidths=0.4)
                  
        m.drawcoastlines(color='dimgray',linewidth=0.7)
                
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
    
    plt.savefig(directoryfigure + 'WACC_Trend_Models_%s_%s-%s.png' % (varnames[v],
                yearmn,yearmx),dpi=300)
    