import matplotlib
matplotlib.use('Agg')


from astropy.io import fits
import glob
import os
import sys
import time 
import h5py
import numpy as np
import pandas as pd 
import subprocess

from scipy import interpolate

import matplotlib.pyplot as plt 
plt.ioff()


from joblib import Parallel, delayed


def get_delay(f_l, f_h, DM):
    '''
    Calculate DM delay between two frequencies for given DM
    Args:
        f_l, f_h: float, low/high frequencies in MHz
        DM: float or int, in pc cm^3
    Returns:
        delay due to DM: float, seconds
    '''
    return 4.15*(10.**3)*(f_l**(-2.)-f_h**(-2.))*DM

def get_SB(freq):
    '''
    Return the central frequency of a LOFAR subband, for  LBA.
    Input: MHz (float) 
    Output: Subband (int)
    '''
    return int(np.floor((1024./200.)*freq))

def get_freq(SB):
    '''
    Return the LOFAR subband of a given frequency, for LBA.
    Input: Subband
    Output: MHz (float) 
    '''
    return (SB/512.)*200./2.

def get_beam(freq):
    beams = np.array([30,35,40,50,55,60,65,70,75,80,85,90])
    freq_to_use = str(beams[np.argsort(np.abs(freq-beams))[0]])

    beam_file = "/home/mkuiack1/AARTFAAC_beamsim/LBAOUTER_AARTFAAC_beamshape_{}MHz.hdf5".format(freq_to_use)
    orig =  np.array(h5py.File(beam_file, mode="r").get('lmbeamintensity_norm'))

    # Make its coordinates; x is horizontal.
    x = np.linspace(0, 2300, orig.shape[1])
    y = np.linspace(0, 2300, orig.shape[0])

    # Make the interpolator function.
    f = interpolate.interp2d(x, y, orig, kind='linear')

    # Construct the new coordinate arrays.
    x_new = np.arange(0, 2300)
    y_new = np.arange(0, 2300)

    # Do the interpolation.
    return f(x_new, y_new)


def get_ddfilelist(filelist, freqs, check_time, DM):
    dd_filelist = []
    for i in range(len(freqs)):
#         print pd.to_datetime(check_time)+pd.to_timedelta(np.round(get_delay(freqs[i],np.max(freqs), DM),0), unit='s')
        try:
            newfile = filelist[str(freqs[i])].loc[pd.to_datetime(check_time)
                                    +pd.to_timedelta(np.round(get_delay(freqs[i],np.max(freqs), DM),0), unit='s') ]
            if str(newfile) == 'nan' :
                newfile = filelist[str(freqs[i])].loc[pd.to_datetime(check_time)
                      +pd.to_timedelta(np.round(get_delay(freqs[i],np.max(freqs), DM),0), unit='s')-
                     +pd.to_timedelta(1, unit='s')]
            dd_filelist.append(newfile)
        except KeyError:
            try:
#                 print(pd.to_datetime(check_time)
#                       +pd.to_timedelta(np.round(get_delay(freqs[i],np.max(freqs), DM),0), unit='s')-
#                      +pd.to_timedelta(1, unit='s'))
                newfile = filelist[str(freqs[i])].loc[pd.to_datetime(check_time)
                      +pd.to_timedelta(np.round(get_delay(freqs[i],np.max(freqs), DM),0), unit='s')-
                     +pd.to_timedelta(1, unit='s')]
                dd_filelist.append(newfile)
            except KeyError:
#                 print "no"
                continue 
    return dd_filelist

def make_dedisperse_image(dd_filelist):
    
    integrated = np.zeros([1024,1024])
    
    for img in dd_filelist:
        hdu_1 = fits.open(img)[0]
#         print os.path.basename(img), np.nanstd(hdu_1.data[0,0,:,:])
        integrated += hdu_1.data[0,0,:,:]
        
    return integrated


# subbands = np.array([281, 284, 287, 291, 294, 298, 301, 304, 
#                      308, 311, 315, 318, 321, 325, 328, 332])

#subbands = np.array([156, 165, 174, 187, 195, 213, 221, 231,
#                     243, 256, 257, 267, 278, 284, 296, 320])


OBS = sys.argv[1] #"202012032122"


subbands = np.array(sorted([os.path.basename(x)[2:5] for x in  glob.glob("/zfs/helios/filer0/mkuiack1/{}/SB*.vis".format(OBS)) ]), dtype=int)

print len(subbands), "subbands in", "/zfs/helios/filer0/mkuiack1/{}/SB*.vis".format(OBS)
if len(subbands) == 0:
    print "/zfs/helios/filer0/mkuiack1/{}/SB*.vis".format(OBS)
    sys.exit()


filelist = pd.DataFrame([])

for i in range(len(subbands)):
    print subbands[i]
    index = pd.to_datetime([os.path.basename(x)[:21]
                            for x in sorted(glob.glob("/zfs/helios/filer0/mkuiack1/{}/*_all/SB".format(OBS)
                                                      +str(subbands[i])
                                                      +"*/imgs/*.fits"))]).round("1s")
    
    df = pd.DataFrame({str(get_freq(subbands[i])):\
                       sorted(glob.glob("/zfs/helios/filer0/mkuiack1/{}/*_all/SB".format(OBS)
                                        +str(subbands[i])
                                        +"*/imgs/*.fits"))},
                      index=index)
    
    filelist = pd.concat([filelist, df.loc[~df.index.duplicated(keep='first')]], axis=1)


# frame = 0
print "Total timesteps:", len(index)

# for fi_ind in index:
    
def make_frame(fi_ind):
    im_stack = np.zeros([2300, 2300])
    f_stack = []
    all_freqs = []
    slice_times = []
    all_std = []
    count = 0 
    psr = []

    for img_file in get_ddfilelist(filelist, get_freq(subbands), fi_ind, 0):

            try:
                data, f=fits.getdata(img_file, header=True)
            except IOError:
                print img_file
                continue  
            if os.path.exists("/hddstore/mkuiack1/png/{}.png".format(f["DATE-OBS"])):
                return 0
            STD = np.std(data[0,0,:,:])


            all_std.append(STD)
    #         print STD
            slice_times.append(f["DATE-OBS"])

            beam_model = get_beam(f[" CRVAL3"]/1e6)
            psr.append(np.mean((data[0,0,:,:]/np.abs(np.median(data[0,0,:,:]))*(np.max(beam_model)/beam_model))[1675:1685,440:450]))

            if STD < 30. and STD > 8.:
                f_stack.append(f)
                all_freqs.append(f[" CRVAL3"])

                im_stack[:,:] += data[0,0,:,:]/np.abs(np.median(data[0,0,:,:]))*(np.max(beam_model)/beam_model)
    
    fig = plt.figure(figsize=(10.8, 10.8))
    ax = plt.subplot(111)
    imgmap = plt.imshow(im_stack, origin="lower",
               vmin=np.mean(im_stack[500:1500,500:1500])-6*np.std(im_stack[500:1500,500:1500]),
               vmax=np.mean(im_stack[500:1500,500:1500])+6*np.std(im_stack[500:1500,500:1500]) )
    plt.text(.01,.98, f["DATE-OBS"], va="top", ha="left",
             fontsize=16, color="white", transform=ax.transAxes,  )
    plt.text(.01,.95, "nSB: {}".format(len(f_stack)), va="top", ha="left",
             fontsize=16, color="white", transform=ax.transAxes,  )


    imgmap.cmap.set_bad('k',1.)
    ax.patch.set_facecolor('k')

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    fig.savefig("/hddstore/mkuiack1/png/{}.png".format(f["DATE-OBS"]), 
                dpi=100,  bbox_inches='tight', facecolor='black')
    plt.close()
    return 0 

_out = Parallel(n_jobs=12)(delayed(make_frame)(fi_ind) for fi_ind in index)


subprocess.call("rsync", "-av", "/hddstore/mkuiack1/png", "/zfs/helios/filer0/mkuiack1/{}/png".format(OBS))

