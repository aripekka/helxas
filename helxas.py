from __future__ import division, print_function
import os

import numpy as np
import matplotlib.pyplot as plt
from silx.io.specfile import SpecFile, SfErrColNotFound

def energy(th,xtal,hkl):
    hc = 1239.842
    if xtal == 'si':
        d = 0.54306
    elif xtal == 'ge':
        d = 0.56574
    refl = np.sqrt(np.sum(hkl[0]**2+hkl[1]**2+hkl[2]**2))
    return hc/(2*d*np.sin(np.radians(th)))*refl

def braggth(energy,xtal,hkl):
    hc = 1239.842
    if xtal == 'si':
        d = 0.54306
    elif xtal == 'ge':
        d = 0.56574
    refl = np.sqrt(np.sum(hkl[0]**2+hkl[1]**2+hkl[2]**2))
    return np.degrees(np.arcsin(hc/(2*d*energy)*refl))

class HelXAS(object):
    '''
    Class for reading and refining the raw data acquired with HelXAS
    '''

    TAU_SCINTILLATOR = 2.1e-6 #deadtime of scintillator in microsecs

    def __init__(self,datafile,datapath='/home/xasadmin/data/',mcadataprefix=None,mcadatasuffix=''):
        self.datapath = datapath
        self.datafile = datafile
        self.mcaprefix = mcadataprefix
        self.mcasuffix = mcadatasuffix

        self.scan_groups = {}
        self.background_fit_order = 2

        self.analyser = None
        self._theta_calibration = 0
        self._energy_calibration = 0
        #self.scan_groups['direct_beam'] = {'signal' : None, 'background' : None}

    @property
    def theta_calibration(self):
        return self._theta_calibration

    @theta_calibration.setter
    def theta_calibration(self,deltath):
        self._theta_calibration = deltath

        if not self.scan_groups == {}:
            sg_key = list(self.scan_groups.keys())[0] #Picks the 'first' scan group key (Python 2 and 3 compatible)
            theta_range = self.scan_groups[sg_key]['signal']['theta']
            theta = (theta_range[0]+theta_range[-1])/2 #calibration at mid scan range
        else:
            theta = 75

        self._energy_calibration = energy(theta+deltath,*self.analyser)-energy(theta,*self.analyser)

    @property
    def energy_calibration(self):
        return self._energy_calibration

    @energy_calibration.setter
    def energy_calibration(self,deltaE):
        self._energy_calibration = deltaE

        if not self.scan_groups == {}:
            sg_key = list(self.scan_groups.keys())[0] #Picks the 'first' scan group key (Python 2 and 3 compatible)
            theta_range = self.scan_groups[sg_key]['signal']['theta']
            theta = (theta_range[0]+theta_range[-1])/2 #calibration at mid scan range
            E0 = energy(theta,*self.analyser)
        else:
            E0 = energy(75,*self.analyser)

        self._theta_calibration = braggth(E0+deltaE,*self.analyser)-energy(E0,*self.analyser)

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
        try:
            theta = specfile[str(scan_numbers[0])+'.1'].data_column_by_name('Bragg angle')
        except:
            #Support for the old theta scan files
            try:
                theta = specfile[str(scan_numbers[0])+'.1'].data_column_by_name('Theta')
            except:
                theta = braggth(specfile[str(scan_numbers[0])+'.1'].data_column_by_name('Energy'),*self.analyser)


        N0_raw = np.zeros(theta.shape) #raw counts
        N0 = np.zeros(theta.shape)     #deadtime corrected signal
        counting_time = np.zeros(theta.shape)   #counting time

        mcano = np.zeros((theta.size,len(scan_numbers)),dtype=int)

        for i in range(len(scan_numbers)):
            ind = str(scan_numbers[i])
            detector = specfile[ind + '.1'].data_column_by_name('Detector')
            seconds = specfile[ind + '.1'].data_column_by_name('Seconds')

            try:
                mcano[:,i] = specfile[ind + '.1'].data_column_by_name('mcano')
            except SfErrColNotFound:
                mcano[:,i] = -np.ones(theta.size)

            N0_raw = N0_raw + detector
            N0 = N0 + self.scintillator_dead_time_correction(detector,seconds)
            counting_time = counting_time + seconds

        scans = {}
        scans['scan_numbers'] = scan_numbers
        scans['theta'] = theta
        scans['mcano'] = mcano

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

    def get_mca(self,sample_str,normalization=None,x_scale = 'energy'):
        '''
        Get the mca data matrix:

        Input:
            sample_str = Either 'direct_beam' or sample_str given for read_I()
            normalization = None, 'transmission'
        '''

        theta = self.scan_groups[sample_str]['signal']['theta']
        mcanos = self.scan_groups[sample_str]['signal']['mcano']

        #Open a mca file to obtain the number of channels
        mca = np.loadtxt(self.mcaprefix + '%05d' % mcanos[0,0] + self.mcasuffix)
        channels = mca.size

        mca_matrix = np.zeros((theta.size,channels))
        mca_err_matrix = np.zeros((theta.size,channels))

        print('Reading MCA. This might take some time..')

        for i in range(mcanos.shape[0]):
            print('Energy ' + str(i+1)+'/'+str(mcanos.shape[0]))
            mca_spectrum = np.zeros((mca.size,))
            mca_err = np.zeros((mca.size,))

            for j in range(mcanos.shape[1]):
                path = self.mcaprefix + '%05d' % mcanos[i,j] + self.mcasuffix
                scan = np.loadtxt(path)
                mca_spectrum = mca_spectrum + scan

            mca_err = np.sqrt(mca_spectrum)
            if normalization == 'transmission':
                I = self.scan_groups[sample_str]['signal']['counts'][i]
                mca_spectrum = mca_spectrum/I
                mca_err = mca_err/I

            mca_matrix[i,:] = mca_spectrum
            mca_err_matrix[i,:] = mca_err

        if x_scale == 'theta':
            return theta+self.theta_calibration, mca_matrix, mca_err_matrix
        else:
            return energy(theta+self.theta_calibration,*self.analyser), mca_matrix, mca_err_matrix
