# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:08:13 2022

@author: Luc Simonin

plotD(TMD[3]+TMD[13]+TMD[18],[],['b','g','r','m'],[800,-0,140,0,0.2,-0.00,0.02,-0.08],['v=1.975','v=1.824','v=1.748'])
plotU(TMU[5]+TMU[2]+TMU[6],[],['b','g','r'],[500,-0,400,0,0.05,-0.00,0.02,-0.08],['v=1.946','v=1.814','v=1.728'])
plotU(TCUI[7],[],['b','g','r'],[80,-80,220,0,0.01,-0.01,0.02,-0.08],['v=1.8'])
plotU(TCUE[9],[],['b','g','r'],[100,-100,220,0,0.0007,-0.0007,0.02,-0.08],['v=1.802'])
plotU(TP[10]+TP[9]+TP[8],[],['b','g','r'],[800,-0,600,0,0.2,-0.000,0.02,-0.08],['No preload','Isotropic preload','Drained shear preload'])
plotISO([ISO[1][0][0:185]]+[ISO[2][0][0:200]]+[ISO[3][0][0:175]],[],['b','g','r'],[80,-80,850,0,0.01,-0.00,0.025,-0.00],['ISO1'+'\n'+'v0=1.974','ISO2'+'\n'+'v0=1.823','ISO3'+'\n'+'v0=1.690'])
"""
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np



# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 18}
font = {'family':'Tahoma','weight':'normal'}

plt.rc('font', **font)


cols=['k','grey','b','g','r','m','orange','c','lime','pink','yellow']
qmaj, qmin = 100, -100
pmaj, pmin = 220, 0
e2maj, e2min = 0.02, -0.02 #deviatoric
e1maj, e1min = 0.03, -0.08 #volumetric
q22, e22 = 120, 0.0001
limits0=[qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]
nr=2
nc=2
names = ['\u03b5'+'$_p$','\u03b5'+'$_q$',"p'",'q']
high=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
legend00 = ['','','','','','','','','','','','','','','','','','','']

def plot1D(data,model,test_col=cols,limits=limits0,legend0=legend00,lab=['a)','b)'],lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(10,8))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[1].plot(e2, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[0].plot(e2, e1, test_col[i],linewidth=lw) 
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes.plot(e1, s1, test_col[i],label=legend0[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes.set_xlim([e1min,e1maj])
    axes.set_ylim([pmin,pmaj])
    axes.set_ylabel('$\it{s1}$',fontsize=fonts+2)
    axes.set_xlabel('\u03b5'+'$_1$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    axes.text(0.01*(e1maj-e1min)+e2min,0.89*(pmaj-pmin)+qmin,lab[1],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)
  


def plotnew(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(2, 2,figsize=(10,7),sharex='col',sharey='row')
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[0,0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[0,1].plot(e2, s2, test_col[i],linewidth=lw)
        axes[1,0].plot(s1, e1, test_col[i],linewidth=lw)
        axes[1,1].plot(e2, e1, test_col[i],linewidth=lw) 
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes[0,0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[0,1].plot(e2, s2, test_col[i],linewidth=lw)
        axes[1,0].plot(s1, e1, test_col[i],linewidth=lw)
        axes[1,1].plot(e2, e1, test_col[i],linewidth=lw)   

    fonts = 12
    ncol0 = ncol
    if (len(data)+len(model))%ncol==0:
        nrow = (len(data)+len(model))//ncol
    else:
        nrow = (len(data)+len(model))//ncol+1
    
    fig.legend(ncol=ncol0, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.5, 0.975-(nrow-1)*0.025),facecolor='whitesmoke')
    
    M = 5
    ntickse1 = ticker.MaxNLocator(M)
    ntickse2 = ticker.MaxNLocator(M)
    nticksp = ticker.MaxNLocator(M)
    nticksq = ticker.MaxNLocator(M)
    
    plt.subplots_adjust(bottom=0.08,top=1.0-nrow*0.05,left=0.1,right=0.95,wspace=0.15, hspace=0.1)
    
    axes[0,0].set_xlim([pmin,pmaj])
    axes[0,0].set_ylim([qmin,qmaj])
    axes[0,0].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[0,0].tick_params(axis='y',labelsize=fonts)
    axes[0,0].yaxis.set_major_locator(nticksq)
    axes[0,0].grid(linestyle=':',linewidth=1)
    axes[0,0].text(0.02*(pmaj-pmin)+pmin,0.92*(qmaj-qmin)+qmin,'a)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

    
    axes[0,1].set_xlim([e2min,e2maj])
    axes[0,1].set_ylim([qmin,qmaj])
    axes[0,1].grid(linestyle=':',linewidth=1)
    axes[0,1].text(0.02*(e2maj-e2min)+e2min,0.92*(qmaj-qmin)+qmin,'b)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)
    
    axes[1,0].set_xlim([pmin,pmaj])
    axes[1,0].set_ylim([e1min,e1maj])
    axes[1,0].set_xlabel('$\it{p}$',fontsize=fonts+2)
    axes[1,0].set_ylabel('\u03b5'+'$_p$',fontsize=fonts+2)
    axes[1,0].tick_params(axis='x',labelsize=fonts)
    axes[1,0].tick_params(axis='y',labelsize=fonts)
    axes[1,0].xaxis.set_major_locator(nticksp)
    axes[1,0].yaxis.set_major_locator(ntickse1)
    axes[1,0].grid(linestyle=':',linewidth=1)
    axes[1,0].text(0.01*(pmaj-pmin)+pmin,0.92*(e1maj-e1min)+e1min,'c)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)
    
    axes[1,1].set_xlim([e2min,e2maj])
    axes[1,1].set_ylim([e1min,e1maj])
    axes[1,1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1,1].tick_params(axis='x',labelsize=fonts)
    axes[1,1].xaxis.set_major_locator(ntickse2)
    axes[1,1].grid(linestyle=':',linewidth=1)
    axes[1,1].text(0.01*(e2maj-e2min)+e2min,0.92*(e1maj-e1min)+e1min,'d)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

def plotD(data,model,test_col=cols,limits=limits0,legend0=legend00,lab=['a)','b)'],lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[1].plot(e2, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[0].plot(e2, e1, test_col[i],linewidth=lw) 
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes[1].plot(e2, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[0].plot(e2, e1, test_col[i],linewidth=lw)  

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes[1].set_xlim([e2min,e2maj])
    axes[1].set_ylim([qmin,qmaj])
    axes[1].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(e2maj-e2min)+e2min,0.89*(qmaj-qmin)+qmin,lab[1],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)
    
    axes[0].set_xlim([e2min,e2maj])
    axes[0].set_ylim([e1min,e1maj])
    axes[0].set_ylabel('\u03b5'+'$_p$',fontsize=fonts+2)
    axes[0].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(e2maj-e2min)+e2min,0.89*(e1maj-e1min)+e1min,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

def plotDv(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5): # to look at specific volume
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[1].plot(e2, s1, test_col[i],label=legend0[i],linewidth=lw)
        axes[0].plot(e2, v, test_col[i],linewidth=lw) 


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes[1].set_xlim([e2min,e2maj])
    axes[1].set_ylim([pmin,pmaj])
    axes[1].set_ylabel('p',fontsize=fonts+2)
    axes[1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(e2maj-e2min)+e2min,0.89*(pmaj-pmin)+pmin,'b)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)
    
    axes[0].set_xlim([e2min,e2maj])
    axes[0].set_ylim([1.67,2.05])
    axes[0].set_ylabel('v',fontsize=fonts+2)
    axes[0].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(e2maj-e2min)+e2min,2.00,'a)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)



def plotU(data,model,test_col=cols,limits=limits0,legend0=legend00,lab=['a)','b)'],lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(e2, s2, test_col[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes[0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(e2, s2, test_col[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes[0].set_xlim([pmin,pmaj])
    axes[0].set_ylim([qmin,qmaj])
    axes[0].set_xlabel('$\it{p}$',fontsize=fonts+2)
    axes[0].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(pmaj-pmin)+pmin,0.89*(qmaj-qmin)+qmin,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

    
    axes[1].set_xlim([e2min,e2maj])
    axes[1].set_ylim([qmin,qmaj])
    axes[1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(e2maj-e2min)+e2min,0.89*(qmaj-qmin)+qmin,lab[1],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

def plotUqp(data,model,test_col=cols,limits=limits0,legend0=legend00,lab=['a)','b)'],lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        qp = [item[4]/item[3] for item in recl]
        v = [item[6] for item in recl]
        axes[0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(e2, qp, test_col[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        qp = [item[4]/item[3] for item in recl]
        i = j+len(data)
        axes[0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(e2, qp, test_col[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes[0].set_xlim([pmin,pmaj])
    axes[0].set_ylim([qmin,qmaj])
    axes[0].set_xlabel('$\it{p}$',fontsize=fonts+2)
    axes[0].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(pmaj-pmin)+pmin,0.89*(qmaj-qmin)+qmin,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

    
    axes[1].set_xlim([e2min,e2maj])
    axes[1].set_ylim([qmin,1.5])
    axes[1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1].set_ylabel('$\it{q}$'+'/'+'$\it{p}$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(e2maj-e2min)+e2min,0.89*(qmaj-qmin)+qmin,lab[1],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)


def plotUeq(data,model,test_col=cols,limits=limits0,legend0=legend00,lab=['a)','b)'],lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    eq = 0.025
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(e2, s2, test_col[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]+eq
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes[0].plot(s1, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(e2, s2, test_col[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes[0].set_xlim([pmin,pmaj])
    axes[0].set_ylim([qmin,qmaj])
    axes[0].set_xlabel('$\it{p}$',fontsize=fonts+2)
    axes[0].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(pmaj-pmin)+pmin,0.89*(qmaj-qmin)+qmin,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

    
    axes[1].set_xlim([e2min,e2maj])
    axes[1].set_ylim([qmin,qmaj])
    axes[1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(e2maj-e2min)+e2min,0.89*(qmaj-qmin)+qmin,lab[1],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

    


def plotISO(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(5,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(s1, e1, test_col[i],label=legend0[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes.plot(s1, e1, test_col[i],label=legend0[i],linewidth=lw)


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.865, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.18,right=0.74,wspace=0.25, hspace=0.1)
    
    axes.set_xlim([pmin,pmaj])
    axes.set_ylim([e1min,e1maj])
    axes.set_ylabel('\u03b5'+'$_p$',fontsize=fonts+2)
    axes.set_xlabel('p',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    
def plotISOloglog(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5): #identify m
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        lne1 = [np.log(e1[i]) for i in range(len(e1))]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        lns1 = [np.log(s1[i]) for i in range(len(s1))]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(lns1, lne1, test_col[i],label=legend0[i],linewidth=lw)
    


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.15,right=0.8,wspace=0.25, hspace=0.1)
    
    # axes.set_xlim([pmin,pmaj])
    axes.set_ylim([-9,-3])
    axes.set_ylabel('ln('+'\u03b5'+'$_p$'+')',fontsize=fonts+2)
    axes.set_xlabel('ln(p)',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    
def plotISOloglogplas(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5): #identify plastic m
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
    fig.tight_layout()
    
    m1 = float(input("What's m for elasticity?"+'\n'))
    kr = float(input("What's kr?"+'\n'))
    pr =100
    
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        lne1 = [np.log(e1[i]) for i in range(len(e1))]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        lns1 = [np.log(s1[i]) for i in range(len(s1))]
        s2 = [item[4] for item in recl]
        spe1 = [e1[i] - (s1[i]/pr)**(1-m1)/((1-m1)*kr) for i in range(len(s1))]
        lnspe1 = [np.log(spe1[i]) for i in range(len(e1))]
        v = [item[6] for item in recl]
        axes.plot(lns1, lnspe1, test_col[i],label=legend0[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.15,right=0.8,wspace=0.25, hspace=0.1)
    
    # axes.set_xlim([pmin,pmaj])
    axes.set_ylim([-9,-3])
    axes.set_ylabel('ln('+'\u03b5'+'$_p$'+')',fontsize=fonts+2)
    axes.set_xlabel('ln(p)',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    
# def plotISOloglog2(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5): #identify kr
#     [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
#     fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
#     fig.tight_layout()
#     m = float(input("What's m?"+'\n'))
#     for i in range(len(data)):
#         recl = data[i]
#         e1 = [item[1] for item in recl]-recl[0][1]
#         e2 = [item[2] for item in recl]-recl[0][2]
#         s1 = [item[3] for item in recl]
#         sps1 = [s1[i]**(1-m) for i in range(len(s1))]
#         s2 = [item[4] for item in recl]
#         v = [item[6] for item in recl]
#         axes.plot(sps1, e1, test_col[i],label=legend0[i],linewidth=lw)
    


#     fonts = 12
   
#     fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
#     M = 5
    
#     plt.subplots_adjust(bottom=0.22,top=0.95,left=0.15,right=0.8,wspace=0.25, hspace=0.1)
    
#     # axes.set_xlim([pmin,pmaj])
#     axes.set_ylim([e1min,e1maj])
#     axes.set_ylabel('\u03b5'+'$_p$',fontsize=fonts+2)
#     axes.set_xlabel('$p^{(1-m)}$',fontsize=fonts+2)
#     axes.tick_params(axis='x',labelsize=fonts)
#     axes.tick_params(axis='y',labelsize=fonts)
#     axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
#     axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
#     axes.grid(linestyle=':',linewidth=1)

def plotISOloglog2(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5): #identify kr and lambda
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
    fig.tight_layout()
    m = float(input("What's m?"+'\n'))
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        sps1 = [1/(1-m)*s1[i]**(1-m) for i in range(len(s1))]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(sps1, e1, test_col[i],label=legend0[i],linewidth=lw)
    


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.25,top=0.95,left=0.15,right=0.8,wspace=0.25, hspace=0.1)
    
    # axes.set_xlim([pmin,pmaj])
    axes.set_ylim([e1min,e1maj])
    axes.set_ylabel('\u03b5'+'$_p$',fontsize=fonts+2)
    axes.set_xlabel('$\\frac{1}{1-m}$'+'$p^{(1-m)}$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    
def plotISOloglog3(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5): #identify lambda
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
    fig.tight_layout()
    m1 = float(input("What's m for elasticity?"+'\n'))
    m2 = float(input("What's m for consolidation?"+'\n'))
    kr = float(input("What's kr?"+'\n'))
    pr=100
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        spe1 = [e1[i] - (s1[i]/pr)**(1-m1)/((1-m1)*kr) for i in range(len(s1))]
        sps1 = [s1[i]**(1-m2) for i in range(len(s1))]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(sps1, spe1, test_col[i],label=legend0[i],linewidth=lw)
    


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.18,right=0.8,wspace=0.25, hspace=0.1)
    
    # axes.set_xlim([pmin,pmaj])
    # axes.set_ylim([e1min,e1maj])
    axes.set_ylabel('\u03b5'+'$_p$'+'$^p$',fontsize=fonts+2)
    axes.set_xlabel('$p^{(1-m)}$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    

def plotISOlog(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        logs1 = [np.log(s1[i]) for i in range(len(s1))]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(s1, -e1, test_col[i],label=legend0[i],linewidth=lw)
    
    for j in range(len(model)):
        "Not updated for model"
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes.plot(s1, e1, test_col[i],label=legend0[i],linewidth=lw)


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.18,right=0.8,wspace=0.25, hspace=0.1)
    
    axes.set_xscale('log')
    axes.set_xlim([pmin,pmaj])
    axes.set_ylim([e1min,e1maj])
    axes.set_ylabel('-'+'\u03b5'+'$_p$',fontsize=fonts+2)
    axes.set_xlabel('p',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)



def plotISOpow(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(6,2.5))
    fig.tight_layout()
    
    mm = float(input('Input the non-linear power m:'))
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        pows1 = [1/(1-mm)*(s1[i]/100)**(1-mm) for i in range(len(s1))]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(pows1, -e1, test_col[i],label=legend0[i],linewidth=lw)
    
    for j in range(len(model)):
        "Not updated for model"
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes.plot(s1, e1, test_col[i],label=legend0[i],linewidth=lw)


    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.25,top=0.95,left=0.18,right=0.8,wspace=0.25, hspace=0.1)
    
    axes.set_xlim([pmin,pmaj])
    axes.set_ylim([e1min,e1maj])
    axes.set_ylabel('-'+'\u03b5'+'$_p$',fontsize=fonts+2)
    axes.set_xlabel(r'$\frac{1}{1-m}$'+'('+r'$\frac{p}{pr}$'+')'+'$^{(1-m)}$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    
    


def plotD3D(data,model,test_col=cols,limits=limits0,legend0=legend00,lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes[1].plot(e2, s2, test_col[i],label=legend0[i],linewidth=lw)
        axes[0].plot(e2, e1, test_col[i],linewidth=lw) 
    
    for j in range(len(model)):
        if input("2 or 3 D? (answer 2 to go 2D, nothing for 3D)\n")=="2":

            recl = model[j]
            e1 = [item[1] for item in recl]-recl[0][1]
            e2 = [item[2] for item in recl]
            s1 = [item[3] for item in recl]
            s2 = [item[4] for item in recl]
            i = j+len(data)
            axes[1].plot(e2, s2, test_col[i],label=legend0[i],linewidth=lw)
            axes[0].plot(e2, e1, test_col[i],linewidth=lw)  
            
        else:
            recl = model[j]
            e1 = [item[1]+item[2]+item[3] for item in recl]-(recl[0][1]+recl[0][2]+recl[0][3])
            e2 = [(2/3*item[1]-1/3*(item[2]+item[3])) for item in recl]
            s1 = [(item[7]+item[8]+item[9])/3 for item in recl]
            s2 = [item[7]-item[9] for item in recl]
            i = j+len(data)
            axes[1].plot(e2, s2, test_col[i],label=legend0[i],linewidth=lw)
            axes[0].plot(e2, e1, test_col[i],linewidth=lw)  

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.25, hspace=0.1)
    
    axes[1].set_xlim([e2min,e2maj])
    axes[1].set_ylim([qmin,qmaj])
    axes[1].set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes[1].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(e2maj-e2min)+e2min,0.89*(qmaj-qmin)+qmin,'b)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)
    
    axes[0].set_xlim([e2min,e2maj])
    axes[0].set_ylim([e1min,e1maj])
    axes[0].set_ylabel('\u03b5'+'$_p$',fontsize=fonts+2)
    axes[0].set_xlabel('\u03b5'+'$_q$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(e2maj-e2min)+e2min,0.89*(e1maj-e1min)+e1min,'a)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)



def plotoe(data,model,v0s,test_col=cols,limits=[1000,0.1,1.05,0.68],legend0=legend00,lw=1.0,ncol=5):
    [pmax,pmin,emax,emin]=limits
    fig, axes = plt.subplots(1, 1,figsize=(5,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]
        s1 = [item[3] for item in recl]
        e = [item[-1]-1 for item in recl]
        axes.plot(s1, e, test_col[i],label=legend0[i],linewidth=lw)
    
    
    for j in range(len(model)):
        v0 = v0s[j]
        recl = model[j]
        cyc = [item[0]/200 for item in recl]
        e1 = [item[1]/3+item[2] for item in recl]-(recl[0][1]/3+recl[0][2])
        e3 = [item[1]/3-item[2]/2 for item in recl]-(recl[0][1]/3-recl[0][2]/2)
        s1 = [item[3]+2/3*item[4] for item in recl]
        s3 = [item[3]-1/3*item[4] for item in recl]
        p = [item[3] for item in recl]
        q = [item[4] for item in recl]
        e = [v0*np.exp(-item[1])-1 for item in recl]
        i = j+len(data)
        axes.plot(s1, e, test_col[i],label=legend0[i],linewidth=lw)

    fonts = 12

    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.875, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.17,right=0.725,wspace=0.25, hspace=0.1)
    
    axes.set_xscale('log')
   
    axes.set_xlim([pmin,pmax])
    # axes.set_ylim([v0-1-0.1,v0-1+0.02])
    axes.set_ylim([emin,emax])
    axes.set_ylabel('void ratio '+'$\it{e}$',fontsize=fonts+2)
    axes.set_xlabel('$\sigma$'+'$_1$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    # axes[0].text(0.01*(e2maj-e2min)+e2min,0.89*(e1maj-e1min)+e1min,'a)',horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)



def plotTCUA_inf(data,model,test_col=cols,limits=[100,0,500,0,0.2,0],legend0=legend00,lab=['a)','b)'],lw=1.0,ncol=5):
    [Nmax,Nmin,pavmax,pavmin,eqmax,eqmin]=limits
    fig, axes = plt.subplots(1, 2,figsize=(10,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        N = [item[0] for item in recl]
        pav = [item[1] for item in recl]
        eqav = [item[2] for item in recl]
        axes[0].plot(N, pav, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(N, eqav, test_col[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        print("length:",len(recl))
        print("rec0",recl[0])
        # N = [k for k in range(len(recl[0]))]
        # pav = [recl[0][k] for k in range(len(recl[0]))]
        # eqav = [recl[1][k] for k in range(len(recl[0]))]
        N = [k for k in range(len(recl))]
        # pav = [recl[0][k] for k in range(len(recl))]
        eqav = [recl[k]-recl[0] for k in range(len(recl))]
        i = j+len(data)
        # axes[0].plot(N, pav, test_col[i],label=legend0[i],linewidth=lw)
        axes[0].plot(N, eqav, test_col[i],label=legend0[i],linewidth=lw)
        axes[1].plot(N, eqav, test_col[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.8,wspace=0.32, hspace=0.1)
    
    axes[0].set_xlim([Nmin,Nmax])
    axes[0].set_ylim([pavmin,pavmax])
    axes[0].set_xlabel('Cycle number',fontsize=fonts+2)
    axes[0].set_ylabel('Average '+'$\it{p}$',fontsize=fonts+2)
    axes[0].tick_params(axis='x',labelsize=fonts)
    axes[0].tick_params(axis='y',labelsize=fonts)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[0].grid(linestyle=':',linewidth=1)
    axes[0].text(0.01*(Nmax-Nmin)+Nmin,0.89*(pavmax-pavmin)+pavmin,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

    
    axes[1].set_xlim([Nmin,Nmax])
    axes[1].set_ylim([eqmin,eqmax])
    axes[1].set_xlabel('Cycle number',fontsize=fonts+2)
    axes[1].set_ylabel('Accumulated '+'$\epsilon$'+'$_q$',fontsize=fonts+2)
    axes[1].tick_params(axis='x',labelsize=fonts)
    axes[1].tick_params(axis='y',labelsize=fonts)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes[1].grid(linestyle=':',linewidth=1)
    axes[1].text(0.01*(Nmax-Nmin)+Nmin,0.89*(eqmax-eqmin)+qmin,lab[1],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

def plotTCUE_inf(data,model,test_col=cols,limits=[100,0,500,0],legend0=legend00,lab=['a)'],lw=1.0,ncol=5):
    [Nmax,Nmin,pavmax,pavmin]=limits
    fig, axes = plt.subplots(1, 1,figsize=(5,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        N = [item[0] for item in recl]
        pav = [item[1] for item in recl]
        axes.plot(N, pav, test_col[i],label=legend0[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        N = [k for k in range(len(recl[0]))]
        pav = [recl[0][k] for k in range(len(recl[0]))]
        i = j+len(data)
        axes.plot(N, pav, test_col[i],label=legend0[i],linewidth=lw)

    fonts = 12
   
    fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.85, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.22,top=0.95,left=0.15,right=0.7,wspace=0.25, hspace=0.1)
    
    axes.set_xlim([Nmin,Nmax])
    axes.set_ylim([pavmin,pavmax])
    axes.set_xlabel('Cycle number',fontsize=fonts+2)
    axes.set_ylabel('Average '+'$\it{p}$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    axes.text(0.01*(Nmax-Nmin)+Nmin,0.89*(pavmax-pavmin)+pavmin,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)


def plotU_stsp(data,model,test_col=cols,limits=limits0,Tit='Title?',lab=['a)'],lw=1.0,ncol=5):
    [qmaj,qmin,pmaj,pmin,e2maj,e2min,e1maj,e1min]=limits
    fig, axes = plt.subplots(1, 1,figsize=(5,2.5))
    fig.tight_layout()
    
    for i in range(len(data)):
        recl = data[i]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]-recl[0][2]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        v = [item[6] for item in recl]
        axes.plot(s1, s2, test_col[i],linewidth=lw)
    
    for j in range(len(model)):
        recl = model[j]
        e1 = [item[1] for item in recl]-recl[0][1]
        e2 = [item[2] for item in recl]
        s1 = [item[3] for item in recl]
        s2 = [item[4] for item in recl]
        i = j+len(data)
        axes.plot(s1, s2, test_col[i],linewidth=lw)

    fonts = 12
   
    # fig.legend(ncol=1, loc='center', prop={'size': fonts-2}, bbox_to_anchor=(0.9, 0.5),facecolor='whitesmoke')
    
    M = 5
    
    plt.subplots_adjust(bottom=0.2,top=0.9,left=0.175,right=0.925,wspace=0.25, hspace=0.1)
    
    plt.title(Tit[0])
    axes.set_xlim([pmin,pmaj])
    axes.set_ylim([qmin,qmaj])
    axes.set_xlabel('$\it{p}$',fontsize=fonts+2)
    axes.set_ylabel('$\it{q}$',fontsize=fonts+2)
    axes.tick_params(axis='x',labelsize=fonts)
    axes.tick_params(axis='y',labelsize=fonts)
    axes.xaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.yaxis.set_major_locator(ticker.MaxNLocator(M))
    axes.grid(linestyle=':',linewidth=1)
    axes.text(0.01*(pmaj-pmin)+pmin,0.89*(qmaj-qmin)+qmin,lab[0],horizontalalignment='left',verticalalignment='bottom',fontsize=fonts+1)

