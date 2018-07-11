from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from silx.io.specfile import SpecFile

datapath = '/home/xasadmin/data/' 

tau = 2.8e-6 #Time constant for scintillator

def energy(th,hkl):
    hc = 1239.842
    d = 0.5431
    refl = np.sqrt(np.sum(hkl[0]**2+hkl[1]**2+hkl[2]**2))
    return hc/(2*d*np.sin(np.radians(th)))*refl

def theta(energy,hkl):
    hc = 1239.842
    d = 0.5431
    refl = np.sqrt(np.sum(hkl[0]**2+hkl[1]**2+hkl[2]**2))
    return np.degrees(np.arcsin(hc/(2*d*energy)*refl))

def deadtime_correction(counts,counting_time,deadtime):
    correction = 1/(1-deadtime*counts/counting_time)
    return counts*correction

 
def quick_xmu(I0_ind,I_ind,dI0_ind=None,dI_ind=None,plt_separate = True,datafile = '1231_Cao_2',dI0_scale=1,dI_scale=1,bg_fit_order=2):

    #open file
    sd = SpecFile(datapath + datafile)

    theta = sd[str(I_ind[0])+'.1'].data_column_by_name('Theta')

    #Load I
    
    I = np.zeros(theta.shape)
    norm = np.zeros(theta.shape)

    for i in I_ind:
        detector = sd[str(i)+'.1'].data_column_by_name('Detector')
        seconds = sd[str(i)+'.1'].data_column_by_name('Seconds')
       
        I = I + deadtime_correction(detector,seconds,tau)
        norm = norm + seconds

    I_err = np.sqrt(I)/norm
    I = I/norm


    #Load I0

    if not I0_ind == None:

        I0 = np.zeros(theta.shape)
        norm = np.zeros(theta.shape)

        for i in I0_ind:
            detector = sd[str(i)+'.1'].data_column_by_name('Detector')
            seconds = sd[str(i)+'.1'].data_column_by_name('Seconds')
           
            I0 = I0 + deadtime_correction(detector,seconds,tau)
            norm = norm + seconds

        I0_err = np.sqrt(I0)/norm
        I0 = I0/norm

    else:
        I0 = np.ones(I.shape)
        I0_err = np.zeros(I.shape)

    #Load dI0

    if not dI0_ind == None:
        theta_dI0 = sd[str(dI0_ind[0])+'.1'].data_column_by_name('Theta')

        dI0 = np.zeros(theta_dI0.shape)
        norm = np.zeros(theta_dI0.shape)

        for i in dI0_ind:
            detector = sd[str(i)+'.1'].data_column_by_name('Detector')
            seconds = sd[str(i)+'.1'].data_column_by_name('Seconds')
           
            dI0 = dI0 + deadtime_correction(detector,seconds,tau)
            norm = norm + seconds

        dI0 = dI0/norm
        dI0 = dI0*dI0_scale #this takes into account the intensity variation of the tube

        pI0 = np.polyfit(theta_dI0,dI0,bg_fit_order)

    else:
        pI0 = np.array([0])

    #Load dI

    if not dI_ind == None:
        theta_dI = sd[str(dI_ind[0])+'.1'].data_column_by_name('Theta')

        dI = np.zeros(theta_dI.shape)
        norm = np.zeros(theta_dI.shape)

        for i in dI_ind:
            detector = sd[str(i)+'.1'].data_column_by_name('Detector')
            seconds = sd[str(i)+'.1'].data_column_by_name('Seconds')

            dI = dI + deadtime_correction(detector,seconds,tau)
            norm = norm + seconds

        dI = dI/norm
        dI = dI*dI_scale #this takes into account the intensity variation of the tube

        pI = np.polyfit(theta_dI,dI,bg_fit_order)

    else:
        pI = np.array([0])

    if plt_separate:
        plt.figure()
        plt.errorbar(theta,I0,I0_err, label = 'I0')
        plt.errorbar(theta,I,I_err, label = 'I')
        if not dI0_ind == None:
            plt.plot(theta_dI0,dI0,'o',label = 'I0 background')
            plt.plot(theta_dI0,np.polyval(pI0,theta_dI0),'r')
        if not dI_ind == None:
            plt.plot(theta_dI,dI,'o',label = 'I background')
            plt.plot(theta_dI,np.polyval(pI,theta_dI),'r')
        plt.xlabel('Theta (deg)')
        plt.ylabel('Normalized intensity')
        plt.legend()

    if not I0_ind == None:
        xmu = -np.log((I-np.polyval(pI,theta))/(I0-np.polyval(pI0,theta)))
    else:
        xmu = -np.log(I-np.polyval(pI,theta))

    if plt_separate:
        plt.figure()
        plt.plot(theta,xmu)
        plt.xlabel('Theta (deg)')
        plt.ylabel('xmu')
        plt.show()
    
    return theta, xmu
