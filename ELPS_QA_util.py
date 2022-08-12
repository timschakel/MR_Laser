#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:36:28 2022

@author: pstijnm2
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

'''
from images get the center of the circles
'''
def process_image_data(img, direction, xco, zco, xcoi, zcoi):
    all_cx = np.zeros(img.shape[direction])
    all_cz = np.zeros(img.shape[direction])
    circz = 0
    circx = 0
    
    for slide in range(img.shape[direction]):
        if direction == 1:
            img_slice = np.squeeze(img[:,slide,:])
        if direction == 2:
            img_slice = np.squeeze(img[:,:,slide])
            
        cz, cx = find_circle(img_slice, xco, zco, xcoi, zcoi)
        
        all_cx[slide] = cx
        all_cz[slide] = cz
        
    return all_cx, all_cz
    
'''
get the isoc from the fitted positions of the rod
'''
def get_isoc(scaler1, scaler2, model):
    return scaler2.inverse_transform(model.predict(scaler1.transform(np.linspace(0.0, 0.0, num=1)[..., None])))

'''
get the slope from the fitter positions of the rod
'''
def get_slope(model):
    return model.coef_[0]
    

'''
from the centers of the circles fit a straight line through them
'''    
def fit_position(x, y, z):
    first_slice, last_slice = find_first_and_last(x)

    x_scaler, y_scaler, z_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x[first_slice:last_slice+1][..., None])
    y_train = y_scaler.fit_transform(y[first_slice:last_slice+1][..., None])
    z_train = z_scaler.fit_transform(z[first_slice:last_slice+1][..., None])

    modelx = HuberRegressor(epsilon=1)
    modelx.fit(y_train, x_train.ravel())

    modelz = HuberRegressor(epsilon=1)
    modelz.fit(y_train, z_train.ravel())
    return x_scaler, y_scaler, z_scaler, modelx, modelz

'''
returns the points of the fitted line (where the rod is)
'''
def get_fit(axis_scanner, scaler1, scaler2, model):
    return scaler2.inverse_transform(model.predict(scaler1.transform(axis_scanner[..., None])))

'''
help function to index only the values where the rods are present
'''
def find_first_and_last(x):
    idx = 0
    for num in x:
        if num != 0.0:
           first = idx
           break
        idx += 1
    
    idx = 0
    for num in x[::-1]:
        if num != 0.0:
            last = x.size - idx - 1
            break
        idx += 1
        
    return first, last

'''
through edge detection and hough transforms find a circle of a specified radius in the image

hough_circle_peaks switches x,y coordinates...

help(hough_circle_peaks):
from skimage import transform, draw
img = np.zeros((120, 100), dtype=int)
radius, x_0, y_0 = (20, 99, 50)
y, x = draw.circle_perimeter(y_0, x_0, radius)
img[x, y] = 1
hspaces = transform.hough_circle(img, radius)
accum, cx, cy, rad = hough_circle_peaks(hspaces, [radius,])
'''
def find_circle(img, xco, zco, xcoi, zcoi):
    f = interp2d(xco, zco, img)

    imgi = f(xcoi,zcoi)
    sizezi,sizexi = imgi.shape
    imgicrop = imgi[int(sizezi/2)-39:int(sizezi/2)+39,int(sizexi/2)-39:int(sizexi/2)+39]

    edges = canny(imgicrop,sigma=5,low_threshold=5,high_threshold=20)
    searchradius = np.arange(10,19)

    hough_res = hough_circle(edges, searchradius)
    accums, cx, cz, radius = hough_circle_peaks(hough_res, searchradius, total_num_peaks=1)
    
    world_cx = xcoi[round(cx[0] + len(xcoi)/2 - 39)] if cx.size > 0 else 0.0
    world_cz = zcoi[round(cz[0] + len(zcoi)/2 - 39)] if cz.size > 0 else 0.0
    
    return world_cz, world_cx

'''
wrapper function for finding the rod in the coronal of sagittal direction 
'''
def find_rod(imdata, direction, axes_1_co, axis_scanner_co, axes_2_co, axes_1_coi, axes_2_coi):
    axis_1, axis_2 = process_image_data(imdata, direction, axes_1_co, axes_2_co, axes_1_coi, axes_2_coi)

    x_scaler, y_scaler, z_scaler, model_axis_1, model_axis_2 = fit_position(axis_1, axis_scanner_co, axis_2)    
    return x_scaler, y_scaler, z_scaler, model_axis_1, model_axis_2, axis_1, axis_2

'''
create the coordinate system of the MRI (normally and interpolated (taken from matlab script))
'''
def get_world_coordinates(dcmInfile, sizex, sizey, sizez):
    # create grid with world coordinates (xco = x-coordinates)
    dx, dy = dcmInfile[0].info.PixelSpacing
    xmin = dcmInfile[0].info.ImagePositionPatient[0]
    xmax = xmin + sizex * dx
    xco = np.arange(xmin,xmax,dx)
    ymin = dcmInfile[0].info.ImagePositionPatient[1]
    ymax = ymin + sizey * dy
    yco = np.arange(ymin,ymax,dy)
    dz = dcmInfile[0].info.SpacingBetweenSlices
    zmin = dcmInfile[0].info.ImagePositionPatient[2]
    zmax = zmin + sizez * dz
    zco = np.arange(zmin,zmax,dz)

    # interpolated sizes (xcoi = x-coordinates interpolated)
    dxi = dx / 10
    dyi = dy / 10
    dzi = dz / 5
    xcoi = np.arange(xmin,xmax,dxi)
    ycoi = np.arange(ymin,ymax,dyi)
    zcoi = np.arange(zmin,zmax,dzi)
    
    return xco, yco, zco, xcoi, ycoi, zcoi
    
'''
make the plot with the found data points and the fitted rod positions
'''
def make_plot(direction, axis_scanner, axis_1, axis_2, scaler_scanner, scaler_laser_1, scaler_laser_2, model_1, model_2):
    first_slice, last_slice = find_first_and_last(axis_1)

    fit_1 = get_fit(axis_scanner[first_slice:last_slice+1], scaler_scanner, scaler_laser_1, model_1)
    fit_2 = get_fit(axis_scanner[first_slice:last_slice+1], scaler_scanner, scaler_laser_2, model_2)
    
    #make plot
    fig, axes = plt.subplots(2,1)
    
    #male subplot 1
    axes[0].scatter(axis_scanner[first_slice:last_slice+1], axis_1[first_slice:last_slice+1],c = 'k', label='data')
    axes[0].plot(axis_scanner[first_slice:last_slice+1], fit_1,'r', label='fitted curve')
    axes[0].set_ylim(-3, 3)
    axes[0].legend()
    if direction == 1:
        axes[0].set_xlabel('y scanner [mm]')
        axes[0].set_ylabel('x laser [mm]')
    elif direction == 2:
        axes[0].set_xlabel('x scanner [mm]')
        axes[0].set_ylabel('y laser [mm]')
    
    #make subplot 2
    axes[1].scatter(axis_scanner[first_slice:last_slice+1], axis_2[first_slice:last_slice+1], c = 'k', label='data')
    axes[1].plot(axis_scanner[first_slice:last_slice+1], fit_2,'r', label='fitted curve')
    axes[1].set_ylim(-3, 3)
    axes[1].legend()
    if direction == 1:
        axes[1].set_xlabel('y scanner [mm]')
        axes[1].set_ylabel('z laser [mm]')
    elif direction == 2:
        axes[1].set_xlabel('x scanner [mm]')
        axes[1].set_ylabel('z laser [mm]')
    
    if direction == 1:
        filename = "ELPS_figure_1.jpg"
        fig.savefig(filename, dpi=160)
    elif direction == 2:
        filename = "ELPS_figure_2.jpg"
        fig.savefig(filename, dpi=160)
    
    return filename
    