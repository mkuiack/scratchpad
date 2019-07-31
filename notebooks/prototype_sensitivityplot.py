
# coding: utf-8

# In[1]:


import glob
import astropy.io.fits as fits
from astropy.time import Time 
import numpy as np
#import matplotlib.pyplot as plt
import datetime as dt
import pickle
import pandas as pd
from scipy.signal import convolve2d
import os
import time
import sys

from joblib import Parallel, delayed

from sourcefinder.accessors import open as open_accessor
from sourcefinder.accessors import sourcefinder_image_from_accessor

from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,match_coordinates_sky,search_around_sky


#plt.rcParams['font.size']=14
#plt.rcParams['axes.labelsize']='large'
#plt.rcParams['axes.titlesize']='large'


# In[2]:


def aperture_pixels(data,x,y,r,meshx,meshy):
    '''Return data (2D array) values which fall within a pixel distance r, of x and y locations. Meshgrids required to 
    number pixels.'''
    return data[np.where(np.sqrt((meshx-x)**2+(meshy-y)**2.) <= r)]

def aperture_bool(x,y,r,meshx,meshy):
    '''Return bool array (2D array) values which fall within a pixel distance r, of x and y locations. Meshgrids required to 
    number pixels.'''
    return (np.sqrt((meshx-x)**2+(meshy-y)**2.) <= r)

def get_lst(image_name):
    return Time(pd.to_datetime(os.path.basename(image_name).split("U")[0],
                        'raise', format="%Y%m%d%H%M%S")).sidereal_time('apparent', 6.868889)


# In[3]:


def distSquared(p0, p1):
    '''
    Calculate the distance between point p0, [x,y], and a list of points p1, [[x0..xn],[y0..yn]]. 
    '''
    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 1.0:
        return np.where(distance == np.min(distance))[0]
    else:
        return None

def pol2cart(rho, phi):
    """
    Polar to Cartesian coordinate conversion, for distance measure around celestial pole.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def compare_flux(sr, catalog_ras, catalog_decs, catalog_fluxs, catalog_flux_errs):
    '''
    Compares the two catalogues, matching sources, and outputs the results of linear fit to the fluxes. 
    '''
    x = []
    y = []

    w = []
    sr_indexes = []
    cat_indexes = []


    for i in range(len(sr)):

        sr_x, sr_y = pol2cart(np.abs(90-sr[i].dec.value),
                np.deg2rad(sr[i].ra.value))

        cat_x, cat_y = pol2cart(np.abs(90-catalog_decs),
                np.deg2rad(catalog_ras))

        index = distSquared((sr_x,sr_y),
                   np.array([cat_x, cat_y]))

        if type(index) == np.ndarray:
            flux = catalog_fluxs[index]
            flux_err = catalog_flux_errs[index]

            cat_indexes.append(index)
            sr_indexes.append(i)
            y.append(float(sr[i].flux))
            x.append(float(flux))
            w.append(float(sr[i].flux.error))
        else:
            continue
            
    if len(x) > 2:
        w = np.array(w,dtype=float)
        fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]

    return fit[0], fit[1]


def process(image_file):
    '''
    Perform an initial quality control filtering step on the incoming image stream. Images
    which are not rejected are then flux calibrated using a reference catalogue.
    '''
    
    print "running process"

    lofarfrequencyOffset = 0.0
    lofarBW = 195312.5
    
    ref_cat = pd.read_csv("/home/kuiack/AARTFAAC_catalogue.csv")
    
    fitsimg = fits.open(image_file)[0]
    
    t = Time(fitsimg.header['DATE-OBS'])
    frq = fitsimg.header['RESTFRQ']
    bw = fitsimg.header['RESTBW']


    # Initial quality condition. 
    if np.nanstd(fitsimg.data[0,0,:,:]) < 1e4:

        # Source find 
        configuration = {
            "back_size_x": 64,
            "back_size_y": 64,
            "margin": 0,
            "radius": 400}

        img_HDU = fits.HDUList(fitsimg)
        imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                                   plane=0),
                                                     **configuration)

        sr = imagedata.extract(det=5.0, anl=3.0,
                               labelled_data=None, labels=[],
                               force_beam=True)

        # Reference catalogue compare
        slope_cor, intercept_cor = compare_flux(sr,
                                       ref_cat["ra"],
                                       ref_cat["decl"],
                                       ref_cat["f_int"],
                                       ref_cat["f_int_err"])

        
  
        fields=[slope_cor, intercept_cor, len(sr)]

        if slope_cor < 1e8:
            return slope_cor, intercept_cor
        else:
            print "bad fit"
            np.nan, np.nan
            return
    else:
        print "bad image"
        return np.nan, np.nan


# In[4]:


aart = pd.read_csv("/home/kuiack/AARTFAAC_catalogue.csv")
ateam = {"ra":np.array([82.88,299.43,350.28,187.07]),"decl":np.array([21.98,40.59,58.54,12.66])}


# In[12]:


with open(sys.argv[1]) as f:
    observations = f.read().splitlines()

image_list = np.array([])
for obs in observations:
        image_list = np.append(image_list,sorted(glob.glob(obs+"/*S2*fits"))[::1000])

print "number of images to process", len(image_list)
#  
#  
def process_image(image): 
#      t1 = time.time()
#  #     print "image", counter, "/", len(image_list[:1000])
#  
#      # image = image_list[1]

    data, header = fits.getdata(image,header=True)
    img = np.copy(data[0,0,:,:])
    wcs = WCS(image)
    LST = Time(header["DATE-OBS"]).sidereal_time('apparent', 6.868889).deg

    meshx, meshy = np.meshgrid(np.linspace(0,1023,1024),np.linspace(0,1023,1024))
    
    im_ra, im_decl, n, nn = wcs.wcs_pix2world(meshx,meshy,1,1,1)
    
    c1 = SkyCoord(np.ravel(im_ra)*u.deg, np.ravel(im_decl)*u.deg, frame='fk5')
    c2 = SkyCoord(aart.ra.values*u.deg, aart.decl.values*u.deg, frame='fk5')

    radius = 3.

    idx1, idx2, sep2d, dist3d = search_around_sky(c1,c2,radius*u.deg)
    im_pix_x, im_pix_y, n, nn = wcs.wcs_world2pix(c1.ra[idx1],c1.dec[idx1],1,1,1)

    img = np.copy(data[0,0,:,:])

    img[np.array(im_pix_y,dtype=int),
        np.array(im_pix_x,dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y+1,0,1023),dtype=int),
        np.array(np.clip(im_pix_x+1,0,1023),dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y-1,0,1023),dtype=int),
        np.array(np.clip(im_pix_x-1,0,1023),dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y+2,0,1023),dtype=int),
        np.array(np.clip(im_pix_x+2,0,1023),dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y-2,0,1023),dtype=int),
        np.array(np.clip(im_pix_x-2,0,1023),dtype=int)] = np.nan


    c3 = SkyCoord(ateam["ra"]*u.deg, ateam["decl"]*u.deg, frame='fk5')

    radius = 10.

    idx1, idx2, sep2d, dist3d = search_around_sky(c1,c3,radius*u.deg)

    im_pix_x,im_pix_y,n,nn = wcs.wcs_world2pix(c1.ra[idx1],c1.dec[idx1],1,1,1)


    img[np.array(im_pix_y,dtype=int),
        np.array(im_pix_x,dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y+1,0,1023),dtype=int),
        np.array(np.clip(im_pix_x+1,0,1023),dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y-1,0,1023),dtype=int),
        np.array(np.clip(im_pix_x-1,0,1023),dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y+2,0,1023),dtype=int),
        np.array(np.clip(im_pix_x+2,0,1023),dtype=int)] = np.nan

    img[np.array(np.clip(im_pix_y-2,0,1023),dtype=int),
        np.array(np.clip(im_pix_x-2,0,1023),dtype=int)] = np.nan


    d90 = np.cos(np.radians(90))*512
    d80 =  np.cos(np.radians(80))*512
    d70 =  np.cos(np.radians(70))*512
    d60 =  np.cos(np.radians(60))*512
    d50 =  np.cos(np.radians(50))*512
    d40 =  np.cos(np.radians(40))*512
    d30 =  np.cos(np.radians(30))*512
    d20 =  np.cos(np.radians(20))*512
    d10 =  np.cos(np.radians(10))*512
    d0 =  np.cos(np.radians(0))*512

    zenith_angle = [d40,d50,d60,d70,d80,d90]
    angles = [np.radians(40),np.radians(50),np.radians(60),np.radians(70),np.radians(80),np.radians(90)]

    x, y = np.meshgrid(np.linspace(0,1024,1024),
                       np.linspace(0,1024,1024))

    sensitivity = []
    area = []
    area_fraction = []

    slope_cor, int_cor = process(image)

    img = (img-int_cor)/slope_cor

    mask = np.zeros((1024,1024), dtype=int)
    for i in range(len(zenith_angle)-1):
        mask = np.zeros((1024,1024), dtype=int)
        mask[ np.where((np.sqrt((x-512)**2.+(y-512)**2.) <= zenith_angle[i]) &
                      (np.sqrt((x-512)**2.+(y-512)**2.) >= zenith_angle[i+1]))]+= 1

        canvas = np.nan*np.ones((1024,1024))
        canvas[np.array(mask, dtype=bool)] = img[np.array(mask, dtype=bool)]
        sensitivity.append(np.nanstd(canvas))
        area.append(np.float(len(np.where(np.isfinite(img[np.array(mask, dtype=bool)]))[0]))/ len(img[np.array(mask, dtype=bool)]) *(np.sin(angles[i+1])-np.sin(angles[i]))*2*np.pi * (180/np.pi)**2.)
	area_fraction.append(np.float(len(np.where(np.isfinite(img[np.array(mask, dtype=bool)]))[0]))/ len(img[np.array(mask, dtype=bool)]))

 #i   area_sum = []

#    idx = np.argsort(sensitivity)
#    area_sum.append(np.array(area)[idx][0])

#    for i in range(len(area)-1):
#        area_sum.append(np.sum(np.array(area)[idx][:i+2]))

    img_result = pd.DataFrame({"sensitivity":sensitivity, 
                               "area":area, "area_fraction":area_fraction, "LST":LST*np.ones(len(area))})

    
    if os.path.isfile("/home/kuiack/AARTFAACsurvey_sensitivity_plot.csv"): 
        previous_result = pd.read_csv("/home/kuiack/AARTFAACsurvey_sensitivity_plot.csv")

        total_result = pd.concat([img_result,previous_result])
        total_result.to_csv("/home/kuiack/AARTFAACsurvey_sensitivity_plot.csv", index=False)
#        print time.time() - t1
    else:
        img_result.to_csv("/home/kuiack/AARTFAACsurvey_sensitivity_plot.csv", index=False)
    return


Parallel(n_jobs=int(sys.argv[2]))(delayed(process_image)(im) for im in image_list)


print "done"


# In[26]:


# image_list = sorted(glob.glob("/data/AS_201809221701/*S2*.fits"))
# counter = 0
# for image in image_list[:100]:
#     t1 = time.time()
# #     print "image", counter

#     # image = image_list[1]

#     data, header =fits.getdata(image,header=True)
#     img = np.copy(data[0,0,:,:])
#     wcs = WCS(image)
    


# In[24]:


# Time(header["DATE-OBS"]).sidereal_time('apparent', 6.868889).deg


# In[27]:


# Time(header["DATE-OBS"]).sidereal_time('apparent', 6.868889).deg


# In[22]:


# tot_survey = pd.read_csv("/home/kuiack/totsensitivity_plot.csv")

# tot_area_sum = []

# idx = np.argsort(tot_survey["sensitivity"])
# tot_area_sum.append(np.array(tot_survey["area_sum"])[idx][0])

# for i in range(len(tot_survey["area_sum"])-1):
# #     idx = np.argsort(sensitivity)
#     tot_area_sum.append(np.sum(np.array(tot_survey["area_sum"])[idx][:i+2]))

# plt.figure(figsize=(6,6))
# plt.plot(np.array(tot_survey["sensitivity"])[idx],1./(np.array(tot_area_sum)))
# plt.xscale("log")
# plt.yscale("log")
# plt.yticks(np.logspace(-12,0,13))
# plt.xlim([1e-3,1e4])
# plt.ylim([1e-12,1e0])
# plt.xlabel("Sensitivity [Jy]")
# plt.ylabel(r"Surface Density [$\mathrm{deg}^{-2}$]")
# plt.tight_layout()

