import numpy as np
import matplotlib.pyplot as plt

import __main__ as _main
_interactive = not hasattr(_main,'__file__')

if not _interactive: # do not run in an interactive session
    # TKinter backend raises error: main thread is not in main loop
    # https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams.update({'figure.max_open_warning': 0})


def perc_clip_val(data,perc=100):
    # return clipping value by percent clip, does not apply clipping
    # data: array to clip
    # perc: percent value
    # output: min/max clip values
    mperc=(100.-perc)*0.01*0.5
    tmp=np.sort(data.flatten())
    minloc=int(len(tmp)*mperc)
    maxloc=len(tmp)-minloc-1
    clipmin=tmp[minloc]
    clipmax=tmp[maxloc]
    return clipmin,clipmax


def perc_clip(data,perc=100):
    # clip data
    # data: array to clip
    # perc=100: percent value, clip min/max (100-perc)/2 percent
    # output: clipped array
    if perc == 100:
        return data
    clipmin,clipmax=perc_clip_val(data,perc)
    return data.clip(clipmin,clipmax)


def plot_vel(vel,h,vmin=None,vmax=None,figsize=[15,4],unit='km/s',tick=np.arange(1.5,6,1)):
    xmax=(vel.shape[0]-1)*h
    zmax=(vel.shape[1]-1)*h
    fs=14
    #fig=plt.figure(figsize=figsize)
    #ax=plt.imshow(vel.transpose(),extent=(0,xmax,zmax,0))
    #plt.tick_params(labelsize=fs)
    #plt.xlabel('Distance (km)',fontsize=fs)
    #plt.ylabel('Depth (km)',fontsize=fs)
    #plt.gca().xaxis.tick_top()
    #plt.gca().xaxis.set_label_position("top")
    ##ax.axes.set_yticks(np.arange(0,zmax+1,1)) 
    fig, ax = plt.subplots(figsize=figsize)
    img=ax.imshow(vel.transpose(),extent=(0,xmax,zmax,0))
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Distance (km)',fontsize=fs)
    ax.set_ylabel('Depth (km)',fontsize=fs)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    #cb=plt.colorbar(shrink=1.0,pad=0.01,aspect=10,ticks=tick)
    cb=fig.colorbar(img,ax=ax,shrink=1.0,pad=0.01,aspect=10,ticks=tick)
    if vmin is None: vmin=vel.min()
    if vmax is None: vmax=vel.max()
    #plt.clim([vmin,vmax])
    img.set_clim([vmin,vmax])
    cb.set_label(unit,fontsize=fs)
    ct=plt.getp(cb.ax,'ymajorticklabels')
    plt.setp(ct,fontsize=fs)
    if not _interactive:
        return fig


def plot_mig(mig,h,figsize=[15,4]):
    xmax=(mig.shape[0]-1)*h
    zmax=(mig.shape[1]-1)*h
    fs=14
    #fig=plt.figure(figsize=figsize)
    #ax=plt.imshow(mig.T,extent=(0,xmax,zmax,0),cmap=plt.cm.gray)
    #plt.xlabel('Distance (km)',fontsize=fs)
    #plt.ylabel('Depth (km)',fontsize=fs)
    #plt.gca().xaxis.tick_top()
    #plt.gca().xaxis.set_label_position("top")
    ##ax.axes.set_yticks(np.arange(0,zmax+1,1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mig.T,extent=(0,xmax,zmax,0),cmap=plt.cm.gray)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Distance (km)',fontsize=fs)
    ax.set_ylabel('Depth (km)',fontsize=fs)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    if not _interactive:
        return fig


def plot_seismo(seismo,dt,perc=100,figsize=[8,8],cmap="gray_r"):
    # plot image traces
    # seismo.shape = nx,nt
    # figsize=[8,8]: matplotlib figure size [inch]
    # perc=100: percent clip
    # cmap="gray_r": matplotlib colormap

    nx,nt = seismo.shape
    plotdata=perc_clip(seismo,perc)
    print("min=%s max=%s"%(plotdata.min(),plotdata.max()))

    xmin, xmax = 0, nx-1
    tmin, tmax = 0, (nt-1)*dt

    fig,ax = plt.subplots(figsize=figsize)
    img=ax.imshow(plotdata.T,aspect='auto',extent=[xmin,xmax,tmax,tmin],cmap=cmap)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Trace number",fontsize='large')
    ax.set_ylabel("Time (s)",fontsize='large')
    if not _interactive:
        return fig
