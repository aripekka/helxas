from __future__ import division, print_function
import os

import numpy as np
import matplotlib.pyplot as plt
from silx.io.specfile import SpecFile

def energy(th,xtal,hkl):
    hc = 1239.842
    if xtal == 'si':
        d = 0.5431
    refl = np.sqrt(np.sum(hkl[0]**2+hkl[1]**2+hkl[2]**2))
    return hc/(2*d*np.sin(np.radians(th)))*refl

def theta(energy,xtal,hkl):
    hc = 1239.842
    if xtal == 'si':
        d = 0.5431
    refl = np.sqrt(np.sum(hkl[0]**2+hkl[1]**2+hkl[2]**2))
    return np.degrees(np.arcsin(hc/(2*d*energy)*refl))

class HelXAS:
    '''
    Class for reading and refining the raw data acquired with HelXAS
    '''

    TAU_SCINTILLATOR = 2.8e-6 #deadtime of scintillator in microsecs

    def __init__(self,datafile,datapath='/home/xasadmin/data/'):
        self.datapath = datapath
        self.datafile = datafile

        self.scan_groups = {}
        self.background_fit_order = 2

        self.analyser = None
        self.theta_calibration = 0
        #self.scan_groups['direct_beam'] = {'signal' : None, 'background' : None}

    def set_analyser(self, crystal_str, hkl):
        '''
        Set the analyser crystal.
        Input:
            crystal_str = only 'si' is now supported
            hkl = either [h,k,l] or 'hkl' or three-digit integer if only single digit indices
        '''

        if type(hkl) == int:
            hkl = str(hkl)
        if not len(hkl) == 3:
            print('ERROR! Invalid or ambiguous reflection!')
            return

        self.analyser = (crystal_str.lower(),(int(hkl[0]),int(hkl[1]),int(hkl[2])))


    def scintillator_dead_time_correction(self, counts, counting_time):
        '''
        Corrects the signal measured with scintillator for the dead time
        '''
        correction = 1/(1-HelXAS.TAU_SCINTILLATOR*counts/counting_time)
        return counts*correction

    def _read_scans(self,specfile,scan_numbers):
        '''
        Internal method to read scans from the specfileself.

        Input: SpecFile instance, list of scan_numbers
        Output: dictionary containing the summed scans
        '''
        theta = specfile[str(scan_numbers[0])+'.1'].data_column_by_name('Theta')

        N0_raw = np.zeros(theta.shape) #raw counts
        N0 = np.zeros(theta.shape)     #deadtime corrected signal
        counting_time = np.zeros(theta.shape)   #counting time

        for i in scan_numbers:
            detector = specfile[str(i)+'.1'].data_column_by_name('Detector')
            seconds = specfile[str(i)+'.1'].data_column_by_name('Seconds')

            N0_raw = N0_raw + detector
            N0 = N0 + self.scintillator_dead_time_correction(detector,seconds)
            counting_time = counting_time + seconds

        scans = {}
        scans['scan_numbers'] = scan_numbers
        scans['theta'] = theta

        scans['counts'] = N0
        scans['raw_counts'] = N0_raw
        scans['counts_error'] = np.sqrt(N0_raw)
        scans['counting_time'] = counting_time

        scans['intensity'] = N0/counting_time
        scans['raw_intensity'] = N0_raw/counting_time
        scans['intensity_error'] = np.sqrt(N0_raw)/counting_time

        return scans

    def set_background_fit_order(self,order,plot=False):

        self.background_fit_order = order

        if plot:
            for key in self.scan_groups:
                th = self.scan_groups[key]['background']['theta']
                I = self.scan_groups[key]['background']['intensity']
                p = np.polyfit(th,I,self.background_fit_order)

                plt.plot(th,I,'o',label=key)
                plt.plot(th,np.polyval(p,th),'k')

            plt.title('Polynomial background fit of order '+str(self.background_fit_order))
            plt.xlabel('Theta (deg)')
            plt.ylabel('Intensity (counts/s)')
            plt.legend()
            plt.show()

    def read_I0(self,signal_scan_numbers,background_scan_numbers,tube_current=10):
        '''
        Loads the measured I0 scans into the memory
        '''
        specfile = SpecFile(os.path.join(self.datapath,self.datafile))
        self.scan_groups['direct_beam'] = {}
        self.scan_groups['direct_beam']['signal'] = self._read_scans(specfile,signal_scan_numbers)
        self.scan_groups['direct_beam']['background'] = self._read_scans(specfile,background_scan_numbers)

        self.scan_groups['direct_beam']['signal']['tube_current'] = tube_current
        self.scan_groups['direct_beam']['background']['tube_current'] = tube_current

    def read_I(self,sample_str,signal_scan_numbers,background_scan_numbers,tube_current=10):
        '''
        Loads the measured I scans into the memory
        '''
        specfile = SpecFile(os.path.join(self.datapath,self.datafile))
        self.scan_groups[sample_str] = {}
        self.scan_groups[sample_str]['signal'] = self._read_scans(specfile,signal_scan_numbers)
        self.scan_groups[sample_str]['background'] = self._read_scans(specfile,background_scan_numbers)

        self.scan_groups[sample_str]['signal']['tube_current'] = tube_current
        self.scan_groups[sample_str]['background']['tube_current'] = tube_current


    def get_spectrum(self,sample_str,x_scale = 'energy'):

        #normalize the signals to the tube current
        direct_beam = self.scan_groups['direct_beam']
        sample = self.scan_groups[sample_str]

        theta = direct_beam['signal']['theta']

        I0 = direct_beam['signal']['intensity']/direct_beam['signal']['tube_current']
        I0_err = direct_beam['signal']['intensity_error']/direct_beam['signal']['tube_current']

        I = sample['signal']['intensity']/sample['signal']['tube_current']
        I_err = sample['signal']['intensity_error']/sample['signal']['tube_current']

        theta_I0bg = direct_beam['background']['theta']
        theta_Ibg = sample['background']['theta']

        I0_bg =  direct_beam['background']['intensity']/direct_beam['background']['tube_current']
        I_bg =  sample['background']['intensity']/sample['background']['tube_current']

        #fit backgrounds
        p0 = np.polyfit(theta_I0bg,I0_bg,self.background_fit_order)
        p = np.polyfit(theta_Ibg,I_bg,self.background_fit_order)

        #compute mux
        mux = -np.log((I-np.polyval(p,theta))/(I0-np.polyval(p0,theta)))
        mux_error = np.sqrt((I0_err/I0)**2 + (I_err/I)**2)

        if x_scale == 'theta':
            return theta+self.theta_calibration, mux, mux_error
        else:
            return energy(theta+self.theta_calibration,*self.analyser), mux, mux_error
