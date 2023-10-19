# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 08:47:22 2022

@author: Luc Simonin

Responding to Guy's comments: trying a simplified version of HySand

Reintroducing shear hardening in the gibbs energy function
One anisotropy
New anisotropy law suggested by Guy f(1-S(dalpha))+(1-f)(1-S(a))
New NCL and lambda
A can vary with p (but just for sake of experiment, fixed to 0 in base model)
A varying with density (log linear, but power can be added for sake of experiment)

No isotropic contraction

Placing the shear hardening inside the yield surface rather than in g

Making the Hardening modulus density dependent

Av option on both A and H

CORRECTION OF LAMBDA IN THE YIELD SURFACE FOR CONSOLIDATION (lam-1/kr)

BETA * n/N

"""
# import autograd.numpy as np
import numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.01,0.04])
check_sig = np.array([60.1,13.1])
check_alp = np.array([[0.05,0.13], [0.09,0.18], [0.02,0.05],
                      [0.01,0.04], [0.005,0.36], [0.02,0.1], [0.1,2.1]])
check_chi = np.array([[9.0,0.1], [10,0.2], [11.0,0.3],
                      [14.0,0.1],[15.0,0.2], [14.0,0.3],[5.0,3.0]])

file = "Dilation and anisotropy"
name = "Dilation and anisotropy file"
mode = 1
ndim = 2
n_y = 3
n_int = 7
n_inp = 1
const =      [  3 ,1.8 , 200 ,50000,40000,0.7, 34   ,  0.8 , 2.057, 1.98  , 1.677 ,0.01,0.002, 60 ,-0.5, 1.5,3.0  ,0.25,10 , 5 ]
name_const = ['NN','v0','pc0', "K" , "G" ,'m',"phic","bmax",'Beta','Gamma','Delta','lb','ld' ,'A0','Ap','Av','ff', 'r' ,'h','b']


def deriv():
    global NN, rN, n_int, n_y
    global v0, pc0
    global pref, kref, gref, m
    global mu
    global bmax
    global Be, Ga, De
    global lb, ld, lam
    global A0
    global r
    NN = int(const[0])
    rN = 1/NN
    n_int = 2*NN+1
    n_y = NN
    v0 = float(const[1])
    pc0 = float(const[2])
    K = float(const[3])
    pref = 100
    kref = K/pref
    G = float(const[4])
    gref = G/pref
    m = float(const[5])
    phic = float(const[6])
    mu = np.tan(phic*np.pi/180)
    bmax = float(const[7])
    Be = float(const[8])
    Ga = float(const[9])
    De = float(const[10])
    lb = float(const[11])
    ld = float(const[12])
    A0 = float(const[13])
    global Ap, Av
    Ap = float(const[14])
    Av = float(const[15])
    global ff
    ff = float(const[16])
    r = float(const[17])
    global h0,b
    h0 = float(const[18])
    b = float(const[19])
    global H0
    H0 = np.zeros([NN])    
    for i in range(NN):
        H0[i]=h0*(1-(i+1)*rN)**b
        
    # Update v0, only works for isotropic consolidation!!!
    ep_ini = ((pc0) / (pref*kref*(1-m)) * ( ((pc0)**2) / pref**2)**(-m/2))
    v0 /= np.exp(-ep_ini)
deriv()

def update(t,eps,sig,alp,chi,dt,deps,dsig,dalp,dchi):
    if alp[-1,0] > 0:
        if sig[1]<=0:
            alp[-1,1] +=1
    alp[-1,0] = sig[1]
    return alp

def pm(sig):
    pm =  ((sig[0])**2 + (kref/(3*gref)) * (1-m) * sig[1]**2 )**(1/2)
    return (pm)

def g(sig,alp):
    temp= - pref / ( kref*(1-m)*(2-m) ) * (pm(sig)/pref)**(2-m) \
          - rN * sig[0] * sum(alp[i,0] for i in range(NN)) \
          - rN * sig[0] * sum(alp[NN+i,0] for i in range(NN)) \
          - rN * sig[1] * sum(alp[i,1] for i in range(NN))
    return (temp)
def dgds(sig,alp):
    temp=-np.array([ ((sig[0]) / (pref*kref*(1-m)) * ( ((sig[0])**2 + kref*(1-m)*sig[1]**2/(3*gref)) / pref**2)**(-m/2))  + rN * sum(alp[i,0] for i in range(2*NN)),
                      (sig[1]/(3*gref*pref) * ( ((sig[0])**2 + kref*(1-m)*sig[1]**2/(3*gref)) / pref**2)**(-m/2))            + rN * sum(alp[i,1] for i in range(NN)) ])
    return (temp)    
def dgda(sig,alp):
    temp = np.zeros([n_int,ndim])
    for i in range(NN):
        temp[i,0] = -rN * sig[0]
        temp[NN+i,0] = -rN * sig[0]
        temp[i,1] = - rN * sig[1]
    return temp
def d2gdsds(sig,alp):
    brack=( ((sig[0])**2 + kref*(1-m)*sig[1]**2 / (3*gref)) / pref**2)
    temp=-np.array([[ (1 / (kref*(1-m)*pref) * brack**(-m/2) - m*(sig[0])**2 / ((1-m)*kref*pref**3) * brack**(-m/2-1))  
                            ,  -m*(sig[0])*sig[1] / (3*gref*pref**3) * brack**(-m/2-1)] ,
                      [-m*(sig[0])*sig[1] / (3*gref*pref**3) * brack**(-m/2-1)
                            ,  1 / (3*gref*pref) * brack**(-m/2) - m*kref*(1-m)*sig[1]**2 / (((3*gref)**2)*pref**3)  * brack**(-m/2-1) ]]) 
    return (temp)
def d2gdsda(sig,alp):
    temp = np.zeros([ndim,n_int,ndim])
    for j in range(NN):
        temp[0,j,0] = - rN
        temp[0,NN+j,0] = - rN
        temp[1,j,1] = - rN
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for j in range(NN):
        temp[j,0,0] = - rN
        temp[NN+j,0,0] = - rN
        temp[j,1,1] = - rN
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    return temp

y_exclude = True

def y(eps,sig,alp,chi):
    temp=np.zeros([n_y])
    pp = ( (sig[0]/pref)**(1-m) - 1 ) / (1-m)
    lnvlratio = (np.log(Be)-(np.log(v0)-eps[0])-lb*pp) / (np.log(Be/De)-(lb-ld)*pp)
    lam = lb - (lb-ld)*lnvlratio
    lnvl = np.log(Be) - lnvlratio*np.log(Be/De)
    DGratio = ( (lnvl-np.log(Ga)) / (np.log(De)-np.log(Ga)) )
    beta_c = bmax * DGratio
    multi = (sig[0]/pref)**Ap * lnvlratio**Av
    A = A0 * multi
    MN_pq = (sig[0]+2/3*sig[1])*(sig[0]-sig[1]/3)
    H = H0 * lnvlratio**Av
    for i in range(NN):
        pc = pc0*(1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m))
        brack = (chi[i,1]*NN \
                 - 3*H[i]*sig[0]*alp[i,1] \
                 - alp[NN,1] * ((i+1)*rN)*beta_c*chi[i,0]*NN \
                + A * ( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) ) * chi[NN,1] )
        temp[i] = brack**2
        temp[i] /=  4 * (((i+1)*rN)*mu)**2 * MN_pq
        temp[i] -= (1 - (NN*chi[NN+i,0]/pc)**r)
    return temp
def dyde(eps,sig,alp,chi):
    tempy=np.zeros([n_y,ndim])
    pp = ( (sig[0]/pref)**(1-m) - 1 ) / (1-m)
    lnvlratio = (np.log(Be)-(np.log(v0)-eps[0])-lb*pp) / (np.log(Be/De)-(lb-ld)*pp)
    dlnvlratiodep = 1 / (np.log(Be/De)-(lb-ld)*pp)
    lam = lb - (lb-ld)*lnvlratio
    dlamdep = - (lb-ld)*dlnvlratiodep
    lnvl = np.log(Be) - lnvlratio*np.log(Be/De)
    dlnvldep = - dlnvlratiodep*np.log(Be/De)
    DGratio = ( (lnvl-np.log(Ga)) / (np.log(De)-np.log(Ga)) )
    dDGratiodep = dlnvldep / (np.log(De)-np.log(Ga))
    beta_c = bmax * DGratio
    dbeta_cdep = bmax * dDGratiodep
    multi = (sig[0]/pref)**Ap * lnvlratio**Av
    dmultidep = (sig[0]/pref)**Ap * Av*dlnvlratiodep * lnvlratio**(Av-1)
    A = A0 * multi
    dAdep = A0 * dmultidep
    MN_pq = (sig[0]+2/3*sig[1])*(sig[0]-sig[1]/3)
    H = H0 * lnvlratio**Av
    dHdep = H0*Av*dlnvlratiodep*lnvlratio**(Av-1)
    for i in range(NN):
        pc = pc0*(1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m))
        dpcdep = pc0 * (1/(1-m))*((-dlamdep*(1-m)/(lam-1/kref)**2)*alp[NN+i,0]) * (1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m)-1)
        brack = (chi[i,1]*NN \
                 - 3*H[i]*sig[0]*alp[i,1] \
                 - alp[NN,1] * ((i+1)*rN)*beta_c*chi[i,0]*NN \
                + A * ( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) ) * chi[NN,1] )
        dbrackdep = (- alp[NN,1] *  ((i+1)*rN)*dbeta_cdep*chi[i,0]*NN \
                 - 3*dHdep[i]*sig[0]*alp[i,1] \
                + dAdep*( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) ) * chi[NN,1] )
        num = brack**2
        den = 4 * (((i+1)*rN)*mu)**2 * MN_pq
        dnumdep = 2 * dbrackdep * brack
        ddendep = 0
        tempy[i,0] = (dnumdep*den-num*ddendep)/den**2
        tempy[i,0] -= (NN*chi[NN+i,0])**r * r*dpcdep* pc**(-r-1)
    return tempy
def dyds(eps,sig,alp,chi):
    tempy=np.zeros([n_y,ndim])
    pp = ( (sig[0]/pref)**(1-m) - 1 ) / (1-m)
    dppdp = (1/pref)**(1-m)/(1-m) * (1-m)*sig[0]**(-m)
    lnvlratio = (np.log(Be)-(np.log(v0)-eps[0])-lb*pp) / (np.log(Be/De)-(lb-ld)*pp)
    dlnvlratiodp = (-lb*dppdp*(np.log(Be/De)-(lb-ld)*pp)+(lb-ld)*dppdp*(np.log(Be)-(np.log(v0)-eps[0])-lb*pp)) / (np.log(Be/De)-(lb-ld)*pp)**2
    lam = lb - (lb-ld)*lnvlratio
    dlamdp = - (lb-ld)*dlnvlratiodp
    lnvl = np.log(Be) - lnvlratio*np.log(Be/De)
    dlnvldp = - dlnvlratiodp*np.log(Be/De)
    DGratio = ( (lnvl-np.log(Ga)) / (np.log(De)-np.log(Ga)) )
    dDGratiodp = dlnvldp / (np.log(De)-np.log(Ga))
    beta_c = bmax * DGratio
    dbeta_cdp = bmax * dDGratiodp
    multi = (sig[0]/pref)**Ap * lnvlratio**Av
    dmultidp = Ap/pref * (sig[0]/pref)**(Ap-1) * lnvlratio**Av \
            + (sig[0]/pref)**Ap * Av*dlnvlratiodp * lnvlratio**(Av-1)
    A = A0 * multi
    dAdp = A0 *dmultidp
    MN_pq = (sig[0]+2/3*sig[1])*(sig[0]-sig[1]/3)
    dMN_pqdp = (2*sig[0]+1/3*sig[1])
    dMN_pqdq = (1/3*sig[0]-4/9*sig[1])
    H = H0 * lnvlratio**Av
    dHdp = H0*Av*dlnvlratiodp*lnvlratio**(Av-1)
    for i in range(NN):
        pc = pc0*(1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m))
        dpcdp = pc0 * (1/(1-m))*(-dlamdp*(1-m)/(lam-1/kref)**2)*alp[NN+i,0] * (1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m)-1)
        brack = (chi[i,1]*NN \
                 - 3*H[i]*sig[0]*alp[i,1] \
                 - alp[NN,1] * ((i+1)*rN)*beta_c*chi[i,0]*NN \
                + A * ( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) ) * chi[NN,1] )
        dbrackdp = (- alp[NN,1] * ((i+1)*rN)*dbeta_cdp*chi[i,0]*NN \
                 - 3*H[i]*alp[i,1] \
                 - 3*dHdp[i]*sig[0]*alp[i,1] \
                + dAdp*( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) )*chi[NN,1] )
        dbrackdq = 0
        num = brack**2
        dem = 4 * (((i+1)*rN)*mu)**2 * MN_pq
        dnumdp = 2 * dbrackdp * brack
        dnumdq = 2 * dbrackdq * brack
        ddemdp = 4 * (((i+1)*rN)*mu)**2 * dMN_pqdp
        ddemdq = 4 * (((i+1)*rN)*mu)**2 * dMN_pqdq
        tempy[i,0] = (dnumdp*dem - num*ddemdp)/dem**2
        tempy[i,1] = (dnumdq*dem - num*ddemdq)/dem**2
        tempy[i,0] -= (NN*chi[NN+i,0])**r * r*dpcdp* pc**(-r-1)
    return tempy
def dyda(eps,sig,alp,chi):
    tempy = np.zeros([n_y,n_int,ndim])
    pp = ( (sig[0]/pref)**(1-m) - 1 ) / (1-m)
    lnvlratio = (np.log(Be)-(np.log(v0)-eps[0])-lb*pp) / (np.log(Be/De)-(lb-ld)*pp)
    lam = lb - (lb-ld)*lnvlratio
    lnvl = np.log(Be) - lnvlratio*np.log(Be/De)
    DGratio = ( (lnvl-np.log(Ga)) / (np.log(De)-np.log(Ga)) )
    multi = (sig[0]/pref)**Ap * lnvlratio**Av
    beta_c = bmax * DGratio
    dbeta_cdap = 0
    A = A0 * multi
    MN_pq = (sig[0]+2/3*sig[1])*(sig[0]-sig[1]/3)
    H = H0 * lnvlratio**Av
    for i in range(NN):
        pc = pc0*(1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m))
        dpcdapN = pc0 * (1/(1-m))*(((1-m)/(lam-1/kref))) * (1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(m/(1-m))
        brack = (chi[i,1]*NN \
                 - 3*H[i]*sig[0]*alp[i,1] \
                 - alp[NN,1] * ((i+1)*rN)*beta_c*chi[i,0]*NN \
                + A * ( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) ) * chi[NN,1] )
        dbrackdaq = - 3*H[i]*sig[0]
        dbrackdapN = - alp[NN,1] * ( ((i+1)*rN)*dbeta_cdap*chi[i,0]*NN )
        dbrackdaqN = -  ((i+1)*rN)*beta_c*chi[i,0]*NN \
                + A*( - (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) )*chi[NN,1]
        num = brack**2
        dnumdaq = 2*dbrackdaq*brack
        dnumdapN = 2*dbrackdapN*brack
        dnumdaqN = 2*dbrackdaqN*brack
        dem = 4 * (((i+1)*rN)*mu)**2 * MN_pq
        ddemdaq = 0
        ddemdapN = 0
        ddemdaqN = 0
        tempy[i,i,1] = (dnumdaq*dem - num*ddemdaq) / dem**2
        tempy[i,NN,1] = (dnumdaqN*dem - num*ddemdaqN) / dem**2
        tempy[i,NN+i,0] = (dnumdapN*dem- num*ddemdapN) / dem**2
        tempy[i,NN+i,0] -= - (NN*chi[NN+i,0])**r * (-r*dpcdapN*pc**(-r-1))
    return tempy
def dydc(eps,sig,alp,chi):
    tempy = np.zeros([n_y,n_int,ndim])
    pp = ( (sig[0]/pref)**(1-m) - 1 ) / (1-m)
    lnvlratio = (np.log(Be)-(np.log(v0)-eps[0])-lb*pp) / (np.log(Be/De)-(lb-ld)*pp)
    lam = lb - (lb-ld)*lnvlratio
    lnvl = np.log(Be) - lnvlratio*np.log(Be/De)
    DGratio = ( (lnvl-np.log(Ga)) / (np.log(De)-np.log(Ga)) )
    beta_c = bmax * DGratio
    multi = (sig[0]/pref)**Ap * lnvlratio**Av
    A = A0 * multi
    MN_pq = (sig[0]+2/3*sig[1])*(sig[0]-sig[1]/3)
    H = H0 * lnvlratio**Av
    for i in range(NN):
        pc = pc0*(1+((1-m)/(lam-1/kref))*alp[NN+i,0])**(1/(1-m))
        brack = (chi[i,1]*NN \
                 - 3*H[i]*sig[0]*alp[i,1] \
                 - alp[NN,1] * ((i+1)*rN)*beta_c*chi[i,0]*NN \
                + A * ( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) ) * chi[NN,1] )
        dbrackdp = - alp[NN,1] * ((i+1)*rN)*beta_c*NN 
        dbrackdq = NN
        dbrackdqN = A*( 1 - alp[NN,1] * (ff*np.sign(alp[NN,1])+(1-ff)*np.sign(chi[i,1])) )
        num = brack**2
        dnumdcp = 2*dbrackdp*brack
        dnumdcq = 2*dbrackdq*brack
        dnumdcqN = 2*dbrackdqN*brack
        dem = 4 * (((i+1)*rN)*mu)**2 * MN_pq
        ddemdcp = 0           
        ddemdcq = 0
        tempy[i,i,0] = (dnumdcp*dem - num*ddemdcp) / dem**2
        tempy[i,i,1] = (dnumdcq*dem - num*ddemdcq) / dem**2
        tempy[i,NN,1] = dnumdcqN / dem
        tempy[i,NN+i,0] = r*chi[NN+i,0]**(r-1) * (NN/pc)**r
    return tempy