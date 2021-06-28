#!/usr/bin/python

import os
import sys
import glob
import h5py
import numpy
import astropy.io.fits as fits
from joblib import Parallel, delayed


def apply_beam(image):
    image_data, header = fits.getdata(image,header=True)
    image_data[0,0,:,:] = image_data[0,0,:,:]*(numpy.max(beam_model)/beam_model)
    fits.writeto(image,image_data,header,overwrite=True)


# dir path for raw iamges
in_image_dir = sys.argv[1]

os.chdir(in_image_dir)
#out_image_dir = in_image_dir+"_beam"

# beam model file from peeyush
beam_file = "/home/kuiack/AARTFAAC_beamsim/LBAOUTER_AARTFAAC_beamshape_60MHz.hdf5"
beam_model = numpy.array(h5py.File(beam_file).get('lmbeamintensity_norm'))

image_list = glob.glob(in_image_dir+"*.fits")
print image_list
#if not os.path.exists("../"+out_image_dir):
#    os.makedirs("../"+out_image_dir)

# open file and multiply by normalized beam model
#for image in image_list:
#    image_data, header = fits.getdata(image,header=True)
#    image_data[0,0,:,:] = image_data[0,0,:,:]*(numpy.max(beam_model)/beam_model)
#    fits.writeto(out_image_dir+image,image_data,header,clobber=True)

Parallel(n_jobs=int(sys.argv[2]))(delayed(apply_beam)(image) for image in image_list)
