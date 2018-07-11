from __future__ import division, print_function
import os

import numpy as np
from silx.io.specfile import SpecFile

class HelXAS:
    '''
    Class for reading and refining the raw data acquired with HelXAS
    '''

    TAU_SCINTILLATOR = 2.8e-6 #deadtime of scintillator in microsecs

    def __init__(self,datafile,datapath='/home/xasadmin/data/'):
        self.datapath = datapath
        self.datafile = datafile

    def scintillator_dead_time_correction(self, counts, counting_time):
        '''
        Corrects the signal measured with scintillator for the dead time
        '''
        correction = 1/(1-HelXAS.TAU_SCINTILLATOR*counts/counting_time)
        return counts*correction

    def read_I0(self,signal_scan_numbers,background_scan_numbers):
        '''
        Loads the measured I0 scans into the memory
        '''

        specfile = SpecFile(os.path.join(self.datapath,self.datafile))

        #Load I0
        theta = specfile[str(signal_scan_numbers[0])+'.1'].data_column_by_name('Theta')

        N0_raw = np.zeros(theta.shape) #raw counts
        N0 = np.zeros(theta.shape)     #deadtime corrected signal
        counting_time = np.zeros(theta.shape)   #counting time

        for i in signal_scan_numbers:
            detector = specfile[str(i)+'.1'].data_column_by_name('Detector')
            seconds = specfile[str(i)+'.1'].data_column_by_name('Seconds')

            N0_raw = N0_raw + detector
            N0 = N0 + self.scintillator_dead_time_correction(detector,seconds)
            counting_time = counting_time + seconds

        self.direct_beam = {}
        self.direct_beam['theta'] = theta

        self.direct_beam['counts'] = N0
        self.direct_beam['raw_counts'] = N0_raw
        self.direct_beam['counts_error'] = np.sqrt(N0_raw)
        self.direct_beam['counting_time'] = counting_time

        self.direct_beam['intensity'] = N0/counting_time
        self.direct_beam['raw_intensity'] = N0_raw/counting_time
        self.direct_beam['intensity_error'] = np.sqrt(N0_raw)/counting_time

        #Load I0 background
        theta_bg = specfile[str(background_scan_numbers[0])+'.1'].data_column_by_name('Theta')

        N0_bg_raw = np.zeros(theta.shape) #raw counts
        N0_bg = np.zeros(theta.shape)     #deadtime corrected signal
        counting_time_bg = np.zeros(theta.shape)   #counting time

        for i in background_scan_numbers:
            detector = specfile[str(i)+'.1'].data_column_by_name('Detector')
            seconds = specfile[str(i)+'.1'].data_column_by_name('Seconds')

            N0_bg_raw = N0_bg_raw + detector
            N0_bg = N0_bg + self.scintillator_dead_time_correction(detector,seconds)
            counting_time_bg = counting_time_bg + seconds

        self.direct_beam_bg = {}
        self.direct_beam_bg['theta'] = theta_bg

        self.direct_beam_bg['counts'] = N0_bg
        self.direct_beam_bg['raw_counts'] = N0_bg_raw
        self.direct_beam_bg['counts_error'] = np.sqrt(N0_bg_raw)
        self.direct_beam_bg['counting_time'] = counting_time_bg

        self.direct_beam_bg['intensity'] = N0_bg/counting_time_bg
        self.direct_beam_bg['raw_intensity'] = N0_bg_raw/counting_time_bg
        self.direct_beam_bg['intensity_error'] = np.sqrt(N0_bg_raw)/counting_time_bg
