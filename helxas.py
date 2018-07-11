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

    def read_I0(self,signal_scan_numbers,background_scan_numbers):
        '''
        Loads the measured I0 scans into the memory
        '''

        specfile = SpecFile(os.path.join(self.datapath,self.datafile))

        self.direct_beam = self._read_scans(specfile,signal_scan_numbers)
        self.direct_beam_bg = self._read_scans(specfile,background_scan_numbers)
