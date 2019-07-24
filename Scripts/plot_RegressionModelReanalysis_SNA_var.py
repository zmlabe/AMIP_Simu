"""
Script calculates regressions on snow cover area (SNA) index for models
AND reanalysis

Notes
-----
    Author : Zachary Labe
    Date   : 24 July 2019
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

### Define time           
now = datetime.datetime.now()
currentmn = str(now.month)
currentdy = str(now.day)
currentyr = str(now.year)
currenttime = currentmn + '_' + currentdy + '_' + currentyr
titletime = currentmn + '/' + currentdy + '/' + currentyr
print('\n' '----Plotting Snow Area Year Regressions - %s----' % titletime)

#### Alott time series
year1 = 1979
year2 = 2015
years = np.arange(year1,year2+1,1)

### Add parameters
ensembles = 10
su = [0,1,2,3,5,6,7]
period = 'DJF' # period for regression
DT = True
varnames = ['SLP','Z500','U200','Z50','T2M','THICK','SST']
runnames = [r'ERA-I',r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']
runnamesm = [r'CSST',r'CSIC',r'AMIP',r'AMQ',r'AMS',r'AMQS']

### Define directories
directoryfigure = '/home/zlabe/Desktop/RegressionSNA_All/'
directorydata = '/home/zlabe/Documents/Research/AMIP/Data/'

def readVar(varnames,runnamesm,period):
    """
    Read in modeled data!
    """
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

###############################################################################

def regressData(x,y,runnamesm):
    """
    Regression function!
    """
    print('\n>>> Using regressData function! \n')
    
    if y.ndim == 5: # 5D array
        slope = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        intercept = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        rvalue = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        pvalue = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        stderr = np.empty((y.shape[0],y.shape[1],y.shape[3],y.shape[4]))
        for model in range(y.shape[0]):
            print('Completed: Regression for %s!' % runnamesm[model])
            for ens in range(y.shape[1]):
                for i in range(y.shape[3]):
                    for j in range(y.shape[4]):
                        ### 1D time series for regression
                        xx = x[model,:]
                        yy = y[model,ens,:,i,j]
                        
                        ### Mask data for nans
                        mask = ~np.isnan(xx) & ~np.isnan(yy)
                        varx = xx[mask]
                        vary = yy[mask]
                        
                        ### Calculate regressions
                        slope[model,ens,i,j],intercept[model,ens,i,j], \
                        rvalue[model,ens,i,j],pvalue[model,ens,i,j], \
                        stderr[model,ens,i,j] = sts.linregress(varx,vary)
                        
    if y.ndim == 4: # 4D array
        slope = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        intercept = np.empty((y.shape[0],y.shape[2],y.shape[3],))
        rvalue = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        pvalue = np.empty((y.shape[0],y.shape[2],y.shape[3],))
        stderr = np.empty((y.shape[0],y.shape[2],y.shape[3]))
        for model in range(y.shape[0]):
            print('Completed: Regression for %s!' % runnamesm[model])
            for i in range(y.shape[2]):
                for j in range(y.shape[3]):
                    ### 1D time series for regression
                    xx = x[model,:]
                    yy = y[model,:,i,j]
                    
                    ### Mask data for nans
                    mask = ~np.isnan(xx) & ~np.isnan(yy)
                    varx = xx[mask]
                    vary = yy[mask]
                        
                    ### Calculate regressions
                    slope[model,i,j],intercept[model,i,j], \
                    rvalue[model,i,j],pvalue[model,i,j], \
                    stderr[model,i,j] = sts.linregress(varx,vary)
                    
    elif y.ndim == 3: #3D array
        slope = np.empty((y.shape[1],y.shape[2]))
        intercept = np.empty((y.shape[1],y.shape[2]))
        rvalue = np.empty((y.shape[1],y.shape[2]))
        pvalue = np.empty((y.shape[1],y.shape[2]))
        stderr = np.empty((y.shape[1],y.shape[2]))
        for i in range(y.shape[1]):
            for j in range(y.shape[2]):
                ### 1D time series for regression
                xx = x[:]
                yy = y[:,i,j]
                
                ### Mask data for nans
                mask = ~np.isnan(xx) & ~np.isnan(yy)
                varx = xx[mask]
                vary = yy[mask]
                        
                ### Calculate regressions
                slope[i,j],intercept[i,j],rvalue[i,j], \
                pvalue[i,j],stderr[i,j] = sts.linregress(varx,vary)
                        
    print('>>> Completed: Finished regressData function!')
    return slope,intercept,rvalue**2,pvalue,stderr

###############################################################################
###############################################################################
###############################################################################
### Regression analysis and plotting    
for rr in range(len(varnames)):    
    ### Read in data from simulations and ERA-Interim
    mod,era,lat,lon = readVar(varnames[rr],runnamesm,period)
    
    ### Read in snow cover area years (Oct-Nov index) (models)
    if DT == True:
        fileindex = 'SNA_Eurasia_ON_DETRENDED.txt'
    elif DT == False:
        fileindex = 'SNA_Eurasia_ON.txt'
    else:
        print(ValueError('WRONG Arguement!'))
    snowdata = np.genfromtxt(directorydata + fileindex,unpack=True,
                             delimiter=',')
    snowarea = snowdata[1:,:] # remove first column of years
    
    ### Read in snow cover area years (Oct-Nov index) (Reanalysis)
    snowareaarea = np.genfromtxt(directorydata + \
                'SNA_Eurasia_ON_ERAi_DETRENDED.txt') # 1D timeseries

    ### Calculate anomalies
    eramean = np.nanmean(era,axis=0)
    modmean = np.nanmean(mod,axis=2)
    
    eraanomq = era - eramean
    modanomq = np.empty((mod.shape))
    for i in range(mod.shape[0]):
        for j in range(mod.shape[1]):
            modanomq[i,j,:,:,:] = mod[i,j,:,:,:,:] - modmean[i,j,:,:,:]

    ### Slice over month(s) of interest
    if period == 'Annual':
        eraanom = np.nanmean(eraanomq[:,:,:,:],axis=1)
        modanom = np.nanmean(modanomq[:,:,:,:,:,:],axis=3)            
    if period == 'OND':
        eraanom = np.nanmean(eraanomq[:,-3:,:,:],axis=1)
        modanom = np.nanmean(modanomq[:,:,:,-3:,:,:],axis=3)
    elif period == 'ND':
        eraanom = np.nanmean(eraanomq[:,-2:,:,:],axis=1)
        modanom = np.nanmean(modanomq[:,:,:,-2:,:,:],axis=3)
    elif period == 'D':
        eraanom = eraanomq[:,-1:,:,:].squeeze()
        modanom = modanomq[:,:,:,-1:,:,:].squeeze()
    elif period == 'F':
        eraanom = eraanomq[:,1,:,:].squeeze()
        modanom = modanomq[:,:,:,1,:,:].squeeze()
    elif period == 'FM':
        eraanom = eraanomq[:,1:3,:,:].squeeze()
        modanom = modanomq[:,:,:,1:3,:,:].squeeze()
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
        
        snowarea = snowarea[:,:-1] # DJF
        snowareaarea = snowareaarea[:-1] #DJF
        
    ### Calculate regression functions
    modcoeff,modint,modr2,modpval,moderr = regressData(snowarea,
                                                       modanom[:,:,:-1,:,:],
                                                       runnamesm)
    eracoeff,eraint,erar2,erapval,eraerr = regressData(snowareaarea,
                                                       eraanom[:-1,:,:],
                                                       runnamesm)
    
    ### Calculate ensemble mean
    modcoeffm = np.nanmean(modcoeff,axis=1) # [model,ens,lat,lon]
            
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Plot low sea ice year regressions
    plt.rc('text',usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
    
    ### Set limits for contours and colorbars
    if varnames[rr] == 'T2M':
        limit = np.arange(-2,2.01,0.05)
        barlim = np.arange(-2,3,1)
        cmap = cmocean.cm.balance
        label = r'\textbf{$^{\circ}$C/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    elif varnames[rr] == 'Z500':
        limit = np.arange(-20,20.1,1)
        barlim = np.arange(-20,21,10)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    elif varnames[rr] == 'Z50':
        limit = np.arange(-50,50.1,1)
        barlim = np.arange(-50,51,25)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/$\bf{10^{6}}$ km$\bf{^{2}}$'
    elif varnames[rr] == 'U200':
        limit = np.arange(-5,5.1,0.25)
        barlim = np.arange(-5,6,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/s/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    elif varnames[rr] == 'U10':
        limit = np.arange(-5,5.1,0.25)
        barlim = np.arange(-5,6,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/s/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    elif varnames[rr] == 'SLP':
        limit = np.arange(-3,3.1,0.25)
        barlim = np.arange(-3,4,3)
        cmap = cmocean.cm.balance
        label = r'\textbf{hPa/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    elif varnames[rr] == 'THICK':
        limit = np.arange(-20,20.1,1)
        barlim = np.arange(-20,21,5)
        cmap = cmocean.cm.balance
        label = r'\textbf{m/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    elif varnames[rr] == 'SST':
        limit = np.arange(-1,1.01,0.05)
        barlim = np.arange(-1,2,1)
        cmap = cmocean.cm.balance
        label = r'\textbf{$^{\circ}$C/$\bf{10^{6}}$ km$\bf{^{2}}$}'
    fig = plt.figure(figsize=(6,5))
    for i in range(len(runnames)):
        if i == 0:
            var = eracoeff
        else:
            var = modcoeffm[i-1,:,:]
        
        ax1 = plt.subplot(3,4,su[i]+1)
        
        if varnames[rr] == 'SST':
            m = Basemap(projection='moll',lon_0=0,resolution='l')   
        else:
            m = Basemap(projection='ortho',lon_0=0,lat_0=89,resolution='l',
                        area_thresh=10000.)
        
        var, lons_cyclic = addcyclic(var, lon)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='white',color='dimgray',
                          linewidth=0.7)
        circle.set_clip_on(False)
        
        cs = m.contourf(x,y,var,limit,extend='both')
                  
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
    
    plt.savefig(directoryfigure + '%s/RegressionSNA_%s_%s.png' % (period,
                                                                   varnames[rr],
                                                                   period),
                                                                    dpi=300)