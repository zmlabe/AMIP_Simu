"""
Script plots trends of 2 m temperature over the WACC period. Subplot compares
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
year1 = 1978
year2 = 2016
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
varnames = ['T2M']
runnames = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

### Call function to read in ERA-Interim
lat,lon,time,lev,era = MOR.readDataR('T2M','surface',False,True)

### Call functions to read in WACCM data
models = np.empty((len(runnames),ensembles,era.shape[0],era.shape[1],
                   era.shape[2],era.shape[3]))
for i in range(len(runnames)):
    lat,lon,time,lev,models[i] = MOM.readDataM('T2M',runnames[i],
                                               'surface',False,True)
    
### Retrieve time period of interest
modq = np.empty((len(runnames),ensembles,era.shape[0]-1,era.shape[2],
                   era.shape[3]))
for i in range(len(runnames)):
    for j in range(ensembles):
        modq[i,j,:,:,:] = UT.calcDecJanFeb(models[i,j,:,:,:],
                                            lat,lon,'surface',1)

eraq = UT.calcDecJanFeb(era,lat,lon,'surface',1)

def detrendData(datavar,years,level,yearmn,yearmx):
    """
    Function removes linear trend

    Parameters
    ----------
    datavar : 4d numpy array or 5d numpy array 
        [ensemble,year,lat,lon] or [ensemble,year,level,lat,lon]
    years : 1d numpy array
        [years]
    level : string
        Height of variable (surface or profile)
    yearmn : integer
        First year
    yearmx : integer
        Last year
    
    Returns
    -------
    datavardt : 4d numpy array or 5d numpy array 
        [ensemble,year,lat,lon] or [ensemble,year,level,lat,lon]
        

    Usage
    -----
    datavardt = detrendData(datavar,years,level,yearmn,yearmx)
    """
    print('\n>>> Using detrendData function! \n')
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Import modules
    import numpy as np
    import scipy.stats as sts

    ### Slice time period
    sliceq = np.where((years >= yearmn) & (years <= yearmx))[0]
    datavar = datavar[:,sliceq,:,:]
    
    ### Detrend data array
    if level == 'surface':
        x = np.arange(datavar.shape[1])
        
        slopes = np.empty((datavar.shape[0],datavar.shape[2],datavar.shape[3]))
        intercepts = np.empty((datavar.shape[0],datavar.shape[2],
                               datavar.shape[3]))
        for ens in range(datavar.shape[0]):
            print('-- Detrended data for ensemble member -- #%s!' % (ens+1))
            for i in range(datavar.shape[2]):
                for j in range(datavar.shape[3]):
                    mask = np.isfinite(datavar[ens,:,i,j])
                    y = datavar[ens,:,i,j]
                    
                    if np.sum(mask) == y.shape[0]:
                        xx = x
                        yy = y
                    else:
                        xx = x[mask]
                        yy = y[mask]
                    
                    if np.isfinite(np.nanmean(yy)):
                        slopes[ens,i,j],intercepts[ens,i,j], \
                        r_value,p_value,std_err = sts.linregress(xx,yy)
                    else:
                        slopes[ens,i,j] = np.nan
                        intercepts[ens,i,j] = np.nan
        print('Completed: Detrended data for each grid point!')
                                
    print('\n>>> Completed: Finished detrendData function!')
    return slopes

def detrendDataR(datavar,years,level,yearmn,yearmx):
    """
    Function removes linear trend from reanalysis data

    Parameters
    ----------
    datavar : 4d numpy array or 5d numpy array 
        [year,month,lat,lon] or [year,month,level,lat,lon]
    years : 1d numpy array
        [years]
    level : string
        Height of variable (surface or profile)
    yearmn : integer
        First year
    yearmx : integer
        Last year
    
    Returns
    -------
    datavardt : 4d numpy array or 5d numpy array 
        [year,month,lat,lon] or [year,month,level,lat,lon]
        
    Usage
    -----
    datavardt = detrendDataR(datavar,years,level,yearmn,yearmx)
    """
    print('\n>>> Using detrendData function! \n')
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Import modules
    import numpy as np
    import scipy.stats as sts
    
    ### Slice time period
    sliceq = np.where((years >= yearmn) & (years <= yearmx))[0]
    datavar = datavar[sliceq,:,:]
    
    ### Detrend data array
    if level == 'surface':
        x = np.arange(datavar.shape[0])
        
        slopes = np.empty((datavar.shape[1],datavar.shape[2]))
        intercepts = np.empty((datavar.shape[1],datavar.shape[2]))
        for i in range(datavar.shape[1]):
            for j in range(datavar.shape[2]):
                mask = np.isfinite(datavar[:,i,j])
                y = datavar[:,i,j]
                
                if np.sum(mask) == y.shape[0]:
                    xx = x
                    yy = y
                else:
                    xx = x[mask]
                    yy = y[mask]
                
                if np.isfinite(np.nanmean(yy)):
                    slopes[i,j],intercepts[i,j], \
                    r_value,p_value,std_err = sts.linregress(xx,yy)
                else:
                    slopes[i,j] = np.nan
                    intercepts[i,j] = np.nan
        print('Completed: Detrended data for each grid point!')

    print('\n>>> Completed: Finished detrendDataR function!')
    return slopes

### Calculate the trend
yearmn = 1995
yearmx = 2014    

modtrend = np.empty((len(runnames),ensembles,models.shape[4],models.shape[5]))
for i in range(len(runnames)):
    modtrend[i,:,:,:] = detrendData(modq[i],years,'surface',yearmn,yearmx)
    print('Completed: Simulation --> %s!' % runnames[i])
    
### Calculate decadal trend
dectrend = modtrend * 10.
    
retrend = detrendDataR(eraq,years,'surface',yearmn,yearmx)

#### Calculate decadal trend
redectrend = retrend * 10.

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for QBO composites
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
MASK = False

### Set limits for contours and colorbars
limit = np.arange(-2,2.01,0.05)
barlim = np.arange(-2,2.01,0.5)
    
fig = plt.figure()
for i in range(len(runnames)):
    var = np.nanmean(dectrend[i],axis=0)
    
    ax1 = plt.subplot(2,3,i+1)
    m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                area_thresh=10000.)
    
    var, lons_cyclic = addcyclic(var, lon)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
    x, y = m(lon2d, lat2d)
              
    m.drawmapboundary(fill_color='white',color='dimgray',linewidth=0.7)
    
    cs = m.contourf(x,y,var,limit,extend='both')
              
    m.drawcoastlines(color='dimgray',linewidth=0.7)
            
    cs.set_cmap(cmocean.cm.balance) 
    ax1.annotate(r'\textbf{%s}' % runnames[i],xy=(0,0),xytext=(0.865,0.90),
                 textcoords='axes fraction',color='k',fontsize=11,
                 rotation=320,ha='center',va='center')

            
###########################################################################
cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{$^{\circ}$C decade$^{-1}$}',
               fontsize=11,color='k',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar.outline.set_edgecolor('dimgrey')

plt.subplots_adjust(wspace=0)
plt.subplots_adjust(hspace=0.01)
plt.subplots_adjust(bottom=0.16)

plt.savefig(directoryfigure + 'WACC_Trend_Models.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################

fig = plt.figure()
var = redectrend

ax1 = plt.subplot(111)
m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
            area_thresh=10000.)

var, lons_cyclic = addcyclic(var, lon)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
x, y = m(lon2d, lat2d)
          
m.drawmapboundary(fill_color='white',color='dimgray',linewidth=0.7)

cs = m.contourf(x,y,var,limit,extend='both')
          
m.drawcoastlines(color='dimgray',linewidth=0.7)
        
cs.set_cmap(cmocean.cm.balance) 
            
###########################################################################
cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)

cbar.set_label(r'\textbf{$^{\circ}$C decade$^{-1}$}',
               fontsize=11,color='k',labelpad=1.4)  

cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))
cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
cbar.outline.set_edgecolor('dimgrey')

plt.subplots_adjust(wspace=0)
plt.subplots_adjust(hspace=0.01)
plt.subplots_adjust(bottom=0.16)

plt.savefig(directoryfigure + 'WACC_Trend_ERAI.png',dpi=300)

print('Completed: Script done!')

for i in range(ensembles):
    fig = plt.figure()
    for j in range(len(runnames)):
        var = dectrend[j,i,:,:]
        
        ax1 = plt.subplot(2,3,j+1)
        m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                    area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
                  
        m.drawmapboundary(fill_color='white',color='dimgray',linewidth=0.7)
        
        cs = m.contourf(x,y,var,limit,extend='both')
                  
        m.drawcoastlines(color='dimgray',linewidth=0.7)
                
        cs.set_cmap(cmocean.cm.balance) 
        ax1.annotate(r'\textbf{%s}' % runnames[j],xy=(0,0),xytext=(0.865,0.90),
                     textcoords='axes fraction',color='k',fontsize=11,
                     rotation=320,ha='center',va='center')
    
                
    ###########################################################################
    cbar_ax = fig.add_axes([0.312,0.1,0.4,0.03])                
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='max',extendfrac=0.07,drawedges=False)
    
    cbar.set_label(r'\textbf{$^{\circ}$C decade$^{-1}$}',
                   fontsize=11,color='k',labelpad=1.4)  
    
    cbar.set_ticks(barlim)
    cbar.set_ticklabels(list(map(str,barlim)))
    cbar.ax.tick_params(axis='x', size=.01,labelsize=8)
    cbar.outline.set_edgecolor('dimgrey')
    
    plt.subplots_adjust(wspace=0)
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(bottom=0.16)

    plt.savefig(directoryfigure + 'WACC_Trend_Models_%s.png' % i,dpi=300)