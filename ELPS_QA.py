#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:02:26 2022

@author: pstijnm2
"""
from __future__ import print_function

__version__ = '20220412'
__author__ = 'pstijnman'

import os
import sys

from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib
from wad_qc.modulelibs import pydicom_series as dcmseries
import numpy as np

import ELPS_QA_util as ELPS

if not 'MPLCONFIGDIR' in os.environ:
    import pkg_resources
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_resources.get_distribution("matplotlib").version.split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

try:
    import pydicom as dicom
except ImportError:
    import dicom
    
#import MRB0_lib as B0

def logTag():
    return "[ELPS_QA_wadwrapper] "

'''
wrapper function to call from the server
'''
def ELPS_QA(data, results, action):
    # read dicom file
    print('Starting ELPS QA')
    dcmInfile = dcmseries.read_files(data.series_filelist[0], True, splitOnPosition=True, readPixelData=False)

    nseries = len(dcmInfile)
    nslices = len(dcmInfile[0]._datasets)
    
    # get image data and dimensions
    imdata1 = dcmInfile[0].get_pixel_array()
    sizez,sizex,sizey = imdata1.shape

    # get coordinate system 
    xco, yco, zco, xcoi, ycoi, zcoi = ELPS.get_world_coordinates(dcmInfile, sizex, sizey, sizez)
    
    #find rod 1
    x_scaler_laser, y_scaler_scanner, z_scaler_laser, modelx_1, modelz_1, x_1, z_1 = ELPS.find_rod(imdata1, 1, xco, yco, zco, xcoi, zcoi)
    isoc_x = ELPS.get_isoc(y_scaler_scanner, x_scaler_laser, modelx_1)
    isoc_z = ELPS.get_isoc(y_scaler_scanner, z_scaler_laser, modelz_1)
    
    #laser vs y scanner
    slope_x = ELPS.get_slope(modelx_1)
    slope_z = ELPS.get_slope(modelz_1)

    #make plot for the first rod
    plt_rod_1 = ELPS.make_plot(1, yco, x_1, z_1, y_scaler_scanner, x_scaler_laser, z_scaler_laser, modelx_1, modelz_1)
    
    #find rod 2
    y_scaler_laser, x_scaler_scanner, z_scaler_laser, modely_2, modelz_2, y_2, z_2 = ELPS.find_rod(imdata1, 2, yco, xco, zco, ycoi, zcoi)
    isoc_y = ELPS.get_isoc(x_scaler_scanner, y_scaler_laser, modely_2)
    isoc_z_2 = ELPS.get_isoc(x_scaler_scanner, z_scaler_laser, modelz_2)
    
    #laser vs x scanner
    slope_y = ELPS.get_slope(modely_2)
    slope_z_2 = ELPS.get_slope(modelz_2)
    
    #make plot for the second rod
    plt_rod_2 = ELPS.make_plot(2, xco, y_2, z_2, x_scaler_scanner, y_scaler_laser, z_scaler_laser, modely_2, modelz_2)
    
    #write results
    results.addObject("Figure1",plt_rod_1)
    results.addObject("Figure2",plt_rod_2)
    
    results.addFloat("ISOC_x [mm]", isoc_x)
    results.addFloat("ISOC_y [mm]", isoc_y)
    results.addFloat("ISOC_z [mm]", isoc_z)
    
    results.addFloat("Helling lateraal [mm/m]", slope_x)
    results.addFloat("Helling longitudinaal [mm/m]", slope_z)
    print('Finished ELPS QA')

'''
taken from the script of tschakel
'''
def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database
    """

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)
    results.addDateTime('AcquisitionDateTime', dt) 
    

if __name__ == "__main__":
    data, results, config = pyWADinput()

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)
            
        elif name == 'ELPS_qa':
            ELPS_QA(data, results, action)

    results.write()