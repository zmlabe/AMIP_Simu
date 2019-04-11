"""
Script calculates anomaly composites of low sea ice years

Notes
-----
    Author : Zachary Labe
    Date   : 13 March 2019
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
print('\n' '----Plotting Low Sea Ice Year Composites - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
period = 'ND'
iceindex = 'ON'
BK = True
varnames = ['SLP','Z500','U200','U10','T2M','THICK','SST']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

### Define directories
if BK == True:
    directoryfigure = '/home/zlabe/Desktop/BK_LowSeaIceYears_%s_index/' % iceindex
else:
    directoryfigure = '/home/zlabe/Desktop/LowSeaIceYears_%s/' % iceindex
directorydata = '/home/zlabe/Documents/Research/AMIP/Data/'

for rr in range(len(varnames)):
    def readVar(varnames,runnamesm,period):
        if varnames == 'SST':
            world = False
        else:
            world = True
        
        ### Call function to read in ERA-Interim (detrended)
        lat,lon,time,lev,era = MOR.readDataR(varnames,'surface',True,world)
        
        ### Call functions to read in WACCM data (detrended)
        models = np.empty((len(runnamesm),ensembles,era.shape[0],era.shape[1],
                           era.shape[2],era.shape[3]))
        for i in range(len(runnamesm)):
            lat,lon,time,lev,models[i] = MOM.readDataM(varnames,runnamesm[i],
                                                       'surface',True,world)
        
        return models,era,lat,lon
    
    ### Read in data from simulations and ERA-Interim
    mod,era,lat,lon = readVar(varnames[rr],runnamesm,period)
    
    ### Read in low sea ice year slices
    if BK == True:
        fileslice = '%s_B-KSeas_SeaIceExtent_1SigmaYears.txt' % iceindex
    else:
        fileslice = '%s_SeaIceExtent_1SigmaYears.txt' % iceindex
    yearslice,iceslice = np.genfromtxt(directorydata + fileslice,
                                       unpack=True)
    yearslice = yearslice.astype(int)
    iceslice = iceslice.astype(int)
    
    ### Calculate anomalies
    eramean = np.nanmean(era,axis=0)
    modmean = np.nanmean(mod,axis=2)
    
    eraanomq = era - eramean
    modanomq = np.empty((mod.shape))
    for i in range(mod.shape[0]):
        for j in range(mod.shape[1]):
            modanomq[i,j,:,:,:] = mod[i,j,:,:,:,:] - modmean[i,j,:,:,:]
            
    if period == 'ND':
        eraanom = np.nanmean(eraanomq[:,-2:,:,:],axis=1)
        modanom = np.nanmean(modanomq[:,:,:,-2:,:,:],axis=3)
    elif period == 'D':
        eraanom = eraanomq[:,-1:,:,:].squeeze()
        modanom = modanomq[:,:,:,-1:,:,:].squeeze()
    elif period == 'F':
        eraanom = eraanomq[:,1,:,:].squeeze()
        modanom = modanomq[:,:,:,1,:,:].squeeze()
    elif period == 'JFM':
        eraanom = np.nanmean(eraanomq[:,0:3,:,:],axis=1)
        modanom = np.nanmean(modanomq[:,:,:,0:3,:,:],axis=3)
    elif period == 'DJF':  
        modanom = np.empty((len(runnamesm),ensembles,modanomq.shape[2]-1,
                         modanomq.shape[4],modanomq.shape[5]))
        for i in range(len(runnamesm)):
            for j in range(ensembles):
                modanom[i,j,:,:,:] = UT.calcDecJanFeb(modanomq[i,j,:,:,:],
                                                    lat,lon,'surface',1)
        eraanom = UT.calcDecJanFeb(eraanomq,lat,lon,'surface',1)
        
        iceslice = iceslice[:-1]
            
    ### Slice low sea ice years
    eralowm = np.nanmean(eraanom[iceslice,:,:],axis=0)
    modlowm = np.nanmean(modanom[:,:,iceslice,:,:],axis=2)
    
    ### Calculate ensemble mean
    modmeanlowm = np.nanmean(modlowm,axis=1)
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Plot low sea ice year composites
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
    ### Set limits for contours and colorbars
    if varnames[rr] == 'T2M':
        limit = np.arange(-2,2.01,0.05)
        barlim = np.arange(-2,3,1)
        cmap = cmocean.cm.balance
        label = r'\textbf{$^{\circ}$C}'
    elif varnames[rr] == 'Z500':
        limit = np.arange(-20,20.1,1)
        barlim = np.arange(-20,21,10)
        cmap = cmocean.cm.balance
        label = r'\textbf{m}'
    elif varnames[rr] == 'Z50':
        limit = np.arange(-50,50.1,1)
        barlim = np.arange(-50,51,25)
        cmap = cmocean.cm.balance
        label = r'\textbf{m}'
    elif varnames[rr] == 'U200':
        limit = np.arange(-5,5.1,0.25)
        barlim = np.arange(-5,6,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/s}'
    elif varnames[rr] == 'U10':
        limit = np.arange(-5,5.1,0.25)
        barlim = np.arange(-5,6,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/s}'
    elif varnames[rr] == 'SLP':
        limit = np.arange(-3,3.1,0.25)
        barlim = np.arange(-3,4,3)
        cmap = cmocean.cm.balance
        label = r'\textbf{hPa}'
    elif varnames[rr] == 'THICK':
        limit = np.arange(-20,20.1,1)
        barlim = np.arange(-20,21,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m}'
    elif varnames[rr] == 'SST':
        limit = np.arange(-1,1.01,0.05)
        barlim = np.arange(-1,2,1)
        cmap = cmocean.cm.balance
        label = r'\textbf{$^{\circ}$C}'
    fig = plt.figure(figsize=(6,5))
    for i in range(len(runnames)):
        if i == 0:
            var = eralowm
    #        pvar = pera
        else:
            var = modmeanlowm[i-1]
    #        pvar = pmodel[i-1]
        
        ax1 = plt.subplot(3,4,su[i]+1)
        
        if varnames[rr] == 'SST':
            m = Basemap(projection='moll',lon_0=0,resolution='l')   
        else:
            m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                        area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    #    pvar,lons_cyclic = addcyclic(pvar, lon)
    #    pvar,lons_cyclic = shiftgrid(180.,pvar,lons_cyclic,start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='white',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs = m.contourf(x,y,var,limit,extend='both')
    #    cs1 = m.contourf(x,y,pvar,colors='None',hatches=['....'],
    #                     linewidths=0.4)
                  
        m.drawcoastlines(color='dimgray',linewidth=0.7)
        if varnames[rr] == 'SST':
            m.fillcontinents(color='dimgray')
                
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
    
    plt.savefig(directoryfigure + '%s/LowSeaIceYears_1_%s_%s.png' % (period,varnames[rr],
                period),dpi=300)