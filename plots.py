import matplotlib.pyplot as pl
import numpy as np


def plot_contours(X,Y,Z,ax,cmap='Grays',xi2_levels=np.array([11.829,6.1801,2.2977,0]),color='r',colormap='Reds',label=r"$\mathrm{Pantheon}$"):
    
    levels = np.max(Z) - xi2_levels/2
    i,j = np.unravel_index(np.argmax(Z), np.array(Z).shape)
    ax.contourf(X,Y,Z,levels=levels,alpha=.5,cmap=colormap)
    ax.text(X[i,j]+0.1,Y[i,j]+0.1,s=label,rotation=45,fontsize=15,color=color)

    return 

    
def plot_cosmo():
    
    fig = pl.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    ax.plot([0,1],[1,0],linestyle='--',c='k')
    ax.plot([0,1.2],[0,0.6],linestyle='--',c='k')
    ax.set_xlabel(r'$\Omega_m$',size=27)
    ax.set_ylabel(r'$\Omega_\Lambda$',size=27)
    fig.text(0.37,0.37,s='Flat universe',rotation=-45,fontsize=20,color='k')
    fig.text(0.2,0.13,s=r'$\mathrm{Expanding \ universe} \ (\Omega_{\Lambda}>\Omega_{m}/2)$',rotation=np.arctan(0.5)*360/(2*np.pi),fontsize=20,color='k')
    
    ax.set_title('oCDM Constrains For SN-only Sample.',size=25)
    ax.set_xlim(0,1.2)
    ax.set_ylim(0,1.2)
    
    return fig,ax
    