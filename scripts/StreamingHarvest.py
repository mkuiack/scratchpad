import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec
import matplotlib.dates as mdates

import tkp.db
import tkp.config
import logging
import csv
import time
import sys
import glob
import itertools


import healpy as hp
import numpy as np
import pandas as pd
import scipy as sp
import datetime
import os

from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.wcs import WCS

import numbers
import math
import scipy
from scipy.stats import norm
from scipy.stats import sem
from scipy import linspace
from scipy import pi,sqrt,exp
from scipy.special import erf

import pylab

from matplotlib.ticker import NullFormatter
from matplotlib.font_manager import FontProperties

plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']='large'
plt.rcParams['axes.titlesize']='large'
pylab.rcParams['legend.loc'] = 'best'
matplotlib.rcParams['text.usetex'] = False


logging.basicConfig(level=logging.INFO)

import warnings
warnings.filterwarnings('ignore')


def nsf(num, n=1):
    """n-Significant Figures"""
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

def num_err(num, err, n=1):
    '''Return number rounded based on error'''
    return np.around(num,int(-(np.floor(np.log10(nsf(err,n=n)))))), nsf(err,n=n)

def clip(data, sigma=3):
    """Remove all values above a threshold from the array.
    Uses iterative clipping at sigma value until nothing more is getting clipped.
    Args:
        data: a numpy array
    """
    data = data[np.isfinite(data)]
    raveled = data.ravel()
    median = np.median(raveled)
    std = np.nanstd(raveled)
    newdata = raveled[np.abs(raveled-median) <= sigma*std]
    if len(newdata) and len(newdata) != len(raveled):
        return clip(newdata, sigma)
    else:
        return newdata

def dump_trans(dbname, dataset_id, engine, host, port, user, pword):
    tkp.db.Database(
        database=dbname, user=user, password=pword,
        engine=engine, host=host, port=port
    )

    # find all the new, candidate transient, sources detected by the pipeline
    transients_query = """
    SELECT  tr.runcat
           ,tr.newsource_type
           ,im.rms_min
           ,im.rms_max
           ,im.detection_thresh
           ,ex.f_int
    FROM newsource tr
         ,image im
         ,extractedsource ex
    WHERE tr.previous_limits_image = im.id
      AND tr.trigger_xtrsrc = ex.id
    """
    
    cursor = tkp.db.execute(transients_query, (dataset_id,))
    transients = tkp.db.generic.get_db_rows_as_dicts(cursor)
    print "Found", len(transients), "new sources"
    return transients

def dump_sources(dbname, dataset_id, engine, host, port, user, pword):
    tkp.db.Database(
        database=dbname, user=user, password=pword,
        engine=engine, host=host, port=port
    )
    # extract the properties and variability parameters for all the running catalogue sources in the dataset
    sources_query = """\
    SELECT  im.taustart_ts
            ,im.tau_time
            ,ex.f_int
            ,ex.f_int_err
            ,ex.f_peak
            ,ex.f_peak_err
            ,ax.xtrsrc
            ,ex.extract_type
            ,ex.det_sigma
            ,ax.runcat as runcatid
            ,ex.ra
            ,ex.decl
            ,ex.ra_err
            ,ex.decl_err
            ,im.band
            ,im.rms_min
            ,im.rms_max
            ,ax.v_int
            ,ax.eta_int
            ,ax.f_datapoints
            ,im.freq_eff
            ,im.url
    FROM extractedsource ex
         ,assocxtrsource ax
         ,image im
         ,runningcatalog rc
    WHERE ax.runcat = rc.id
      AND ax.xtrsrc = ex.id
      and ex.image = im.id
      AND rc.dataset = %s
      ORDER BY rc.id
    """
    cursor = tkp.db.execute(sources_query, (dataset_id,))
    sources = tkp.db.generic.get_db_rows_as_dicts(cursor)

    print "Found", len(sources), "source datapoints"

    return sources 

def distSquared(p0, p1):
    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 3.0:
        return np.where(distance == np.min(distance))[0]
    else:
        return None
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def retrieve_source(full_data, run_id):
    source_df = full_data[(full_data.runcatid == run_id)]
    
#     source_df = source_df.groupby('taustart_ts', as_index=False)
    source_df.set_index(source_df.taustart_ts, inplace=True)
    return source_df.sort_index()


def plot_lightcurve(full_data, run_id, ion_sub=False, roll_len = 1*60, roll_type = 'triang', stdout=True):

    source_df = full_data[(full_data.runcatid == run_id)]
    if stdout:
        print source_df.wm_ra.iloc[0], source_df.wm_decl.iloc[0]

    pd.to_datetime(source_df.taustart_ts)
    source_df = source_df.groupby('taustart_ts', as_index=False).mean()
    source_df.set_index(source_df.taustart_ts, inplace=True)

    if ion_sub:
        rolling = source_df.f_int.rolling(roll_len, win_type=roll_type)
        source_df.f_int = source_df.f_int-rolling.mean()


    plt.rcParams['font.size']=16
    plt.rcParams['axes.labelsize']='large'
    plt.rcParams['axes.titlesize']='large'
    pylab.rcParams['legend.loc'] = 'best'

    ylim = [np.nanmean(source_df.f_int)-6.0*np.nanstd(source_df.f_int),
            np.nanmean(source_df.f_int)+10.0*np.nanstd(source_df.f_int)]



    myFmt = mdates.DateFormatter('%H:%M')
    source_df["taustart_ts"] = pd.to_datetime(source_df["taustart_ts"])
    obs_dates = np.unique([100*x.month+x.day for x in source_df["taustart_ts"]])

    n_hours = np.array([]) 
    for i in obs_dates:
        index = (100*pd.DatetimeIndex(source_df["taustart_ts"]).month+pd.DatetimeIndex(source_df["taustart_ts"]).day == i)# & (source_df.extract_type == 0)
        n_hours = np.append(n_hours, len(np.unique(pd.DatetimeIndex(source_df["taustart_ts"][index]).hour)))
    hour_ratio = [i/n_hours.sum() for i in n_hours ]
    gs_ratio = np.append((hour_ratio)/min(hour_ratio),1)

    gs = gridspec.GridSpec(1, len(obs_dates)+1, width_ratios=gs_ratio) 

    figcount = 0
    figure = plt.figure(figsize=(4*len(obs_dates),6))

    for i in obs_dates:
        index = (100*pd.DatetimeIndex(source_df["taustart_ts"]).month+pd.DatetimeIndex(source_df["taustart_ts"]).day == i)# & (source_df.extract_type == 0)
        ax = plt.subplot(gs[figcount])
        ax.locator_params(nticks=6)
        ax.errorbar(source_df["taustart_ts"].values[index],
                    source_df["f_int"].values[index],
                    yerr=source_df["f_int_err"].values[index],
                    fmt=".",c="#1f77b4",ecolor="#ff7f0e")

        if figcount > 0:
            ax.set_yticks([])
        if figcount ==0:
            plt.ylabel("Flux [Jy]")
            ax.yaxis.set_ticks_position('left')
        if stdout:
            print source_df["taustart_ts"].values[index][0]
        plt.annotate("{}-{}".format(pd.DatetimeIndex(source_df["taustart_ts"].values[index]).day[0],
                                    pd.DatetimeIndex(source_df["taustart_ts"].values[index]).month[0]),
                                    xy=(0.95,0.95), xycoords='axes fraction',
                                    horizontalalignment='right', verticalalignment='top',fontsize=16)

        plt.xticks(rotation=90)
        ax.set_ylim(ylim)
        ax.xaxis.set_major_formatter(myFmt)
        figcount+=1

    hist_index = np.isfinite(source_df["f_int"]) #& (source_df.extract_type == 0)
    plt.subplot(gs[figcount])
    (mu, sigma) = norm.fit(source_df["f_int"].iloc[hist_index].values)
    n, bins, patches   =  plt.hist(source_df["f_int"].values[hist_index],
                                   bins=100,normed=1, orientation='horizontal',facecolor="#1f77b4")
    y = mlab.normpdf( bins, mu, sigma)
    if stdout:
        print "Gaus fit: mu {}, sigma {}".format(round(mu,3),round(sigma,3))
    
    l = plt.plot(y,bins,'r--', linewidth=2)
    # plt.title("Source: N = {}".format(len(source_df["f_int"].values[np.isfinite(source_df["f_int"])])))
    plt.annotate("Total N:\n{}".format(len(source_df["f_int"].values[hist_index])),
                                xy=(0.95,0.95), xycoords='axes fraction',
                                horizontalalignment='right', verticalalignment='top',fontsize=16)
    # plt.ylabel("Normalized N")
    plt.yticks([])
    plt.ylim(ylim)
    plt.xticks(rotation=90)

    plt.subplots_adjust(wspace=0.1, hspace=0)
    return figure
#     plt.show()
# fig.text(0.5, 0.04, 'date', ha='center')
# plt.tight_layout()
# print(source_df["wm_ra"].values[0],source_df["wm_decl"].values[1])
# plt.savefig("{}_multiday_lightcurve.png".format(key))

def run_query(transients_query, dbname, dataset_id, engine, host, port, user, pword):
    tkp.db.Database(
        database=dbname, user=user, password=pword,
        engine=engine, host=host, port=port
    )

    cursor = tkp.db.execute(transients_query, (dataset_id,))
    transients = tkp.db.generic.get_db_rows_as_dicts(cursor)
    return transients


engine = 'postgresql'
host = 'localhost'
port = 5432
user = 'mkuiack'
password = 'Morecomplicatedpass1234'

query_loglevel = logging.WARNING  # Set to INFO to see queries, otherwise WARNING

tot_timer= time.time()

vlssr = pd.read_csv("/home/kuiack/VLSSr.tsv", comment="#", delimiter="\t")
# tgss = pd.read_csv("/home/kuiack/TGSSADR1_7sigma_catalog.tsv", delimiter="\t")
aart = pd.read_csv("/home/kuiack/AARTFAAC_catalogue.csv")
ateam = {"ra":np.array([82.88,299.43,350.28,187.07]),
         "decl":np.array([21.98,40.59,58.54,12.66])}


AS_db = sys.argv[1]
dataset = 1

logging.getLogger('sqlalchemy.engine').setLevel(query_loglevel)


# Catalogue filter items
aart_coord = SkyCoord(aart.ra.values*u.deg, aart.decl.values*u.deg, frame='fk5')
ateam_coord = SkyCoord(ateam["ra"]*u.deg, ateam["decl"]*u.deg, frame='fk5')

# Inspection plot making items
stamp_side = 600
half_side = int(stamp_side/2)

x = np.linspace(0, stamp_side, stamp_side)
y = np.linspace(0, stamp_side, stamp_side)
x, y = np.meshgrid(x, y)

map_dir = "/home/kuiack/skymaps/"
files = glob.glob(map_dir+"*.fits")
BANDS = [os.path.basename(i)[:4] for i in files]

delta = 100 
vlssr_thresh= 5

map_load_1 = hp.fitsfunc.read_map(map_dir+BANDS[2]+"_512_map.fits")
map_load_2 = hp.fitsfunc.read_map(map_dir+BANDS[3]+"_512_map.fits")
map_load_3 = hp.fitsfunc.read_map(map_dir+BANDS[18]+"_512_map.fits")



# get beginning and end time
time_1 = time.time()
db = tkp.db.Database(engine=engine, host=host, port=port,
                     user=user, password=password, database=AS_db)

db.connect()
session = db.Session()

transients_query = """
    (SELECT taustart_ts 
    FROM image ORDER BY taustart_ts ASC LIMIT 30)
    UNION ALL
    (SELECT taustart_ts 
    FROM image ORDER BY taustart_ts DESC LIMIT 30)
    """

time_range = pd.DataFrame(run_query(transients_query, AS_db, dataset, 
                                          engine, host, port, user, password))

db._configured = False
del db, session

starttime = np.min(time_range[np.abs(time_range.diff(periods= -10)
                                                               .astype('timedelta64[s]')) < 15.]
                                             .taustart_ts) 
endtime = np.max(time_range[np.abs(time_range.diff(periods= -10)
                                                             .astype('timedelta64[s]')) < 15.]
                                           .taustart_ts)


time_intervals = pd.date_range(start= starttime,
                               end= endtime + datetime.timedelta(minutes=9), freq="10min")

tot_time = len(pd.date_range(start=starttime,end=endtime, freq="1s"))

print "Get time intervals:",  time.time() - time_1

fig_n = 1
tot_candidate = 0
tot_filtered = 0

for t1, t2 in zip(time_intervals[:-1],time_intervals[1:]):
    time_1 = time.time()
    db = tkp.db.Database(engine=engine, host=host, port=port,
                         user=user, password=password, database=AS_db)

    db.connect()
    session = db.Session()


# Works, but is slow.
    transients_query = """
        SELECT  im.taustart_ts
                ,im.freq_eff
                ,im.band
                ,im.rms_min
                ,im.rms_max
                ,im.url
                ,ex.f_int
                ,ex.f_int_err
                ,ex.f_peak
                ,ex.f_peak_err
                ,ex.ra
                ,ex.decl
                ,ex.ra_err
                ,ex.decl_err
                ,ex.extract_type
                ,ex.det_sigma
                ,ax.runcat as runcatid
                ,ax.v_int
                ,ax.eta_int
                ,ax.f_datapoints
        FROM extractedsource ex
             ,assocxtrsource ax
             ,image im
             ,runningcatalog rc
        WHERE ax.runcat = rc.id 
           AND ax.xtrsrc = ex.id
           AND ex.image = im.id
           AND rc.id IN
        (SELECT DISTINCT ns.runcat 
        FROM image im, newsource ns 
        WHERE ns.trigger_xtrsrc 
        IN (SELECT ex.id 
            FROM extractedsource ex, image im 
            WHERE ex.image = im.id 
            AND im.taustart_ts 
            BETWEEN '{}' AND '{}'))""".format(t1, t2)

    transients = pd.DataFrame(run_query(transients_query, AS_db, dataset, 
                              engine, host, port, user, password))

    print "Run Query:", t1,t2, time.time() - time_1	
    if len(transients) == 0:
	print "no runcats in time interval"
	continue 

    transients = transients[transients.rms_max < 3e2]
    

    db._configured = False
    del db, session


    # Reduce with conditions requireing simultaneous 5sigma and one 8sigma detection
    time_1 = time.time()

    N_detections = 0 

    reduced = pd.DataFrame([])


    transients["round_times"] = transients.taustart_ts.dt.round("1s")


    for _id in np.unique(transients.runcatid):
	    band_timediff = (np.min(transients[(transients.runcatid == _id) & \
        	                 (transients.band == 23) & \
                	         (transients.extract_type == 0)].round_times) - np.min(transients[(transients.runcatid == _id) & \
                      	(transients.band == 24) & \
          	            (transients.extract_type == 0)].round_times)).total_seconds()

	    if band_timediff >= 0 and band_timediff <= 600 \
	        and (np.max(transients[(transients.runcatid == _id )].det_sigma) > 8 ):
	
        	    if len(reduced) == 0:
		                reduced = pd.DataFrame(transients[(transients.runcatid == _id)])
        	    else:
        	        reduced = pd.concat([reduced, transients[(transients.runcatid == _id)]])
    print "Sigma filter:", time.time() - time_1
    if len(reduced) == 0:
	print "no valid transients in time interval"
        continue
 
# Make small reference catalogue of candidates
    base = reduced.groupby("runcatid").median()
    base["taustart_ts"] = reduced.groupby("runcatid").first().taustart_ts
    base["timestep"] = [x.timestamp() for x in base.taustart_ts]

# AARTFAAC Catalogue, Ateam, airplane filter 
    time_1 = time.time()


    filtered = pd.DataFrame([], columns=base.keys())
    AART_catsource = pd.DataFrame([], columns=base.keys())


    for i in base.index:
        try:
            c1 = SkyCoord(base.loc[i].ra*u.deg, base.loc[i].decl*u.deg, frame='fk5')

            c2 = SkyCoord(base.drop(index=i).ra.values*u.deg, 
                      base.drop(index=i).decl.values*u.deg, frame='fk5')
        except IndexError:
            print i
        if np.min(c1.separation(ateam_coord).deg) < 10:
            continue 

        elif np.min(c1.separation(aart_coord).deg) < 3 and base.loc[i].f_datapoints.astype(float) > 10:
            if len(AART_catsource) == 0:
                AART_catsource = pd.DataFrame(base.loc[i]).T
            else:
                AART_catsource = pd.concat([AART_catsource, pd.DataFrame(base.loc[i]).T])


        elif np.logical_or(((c1.separation(c2).deg) > 5),
                           (np.abs(base.loc[i].timestep - base.drop(index=i).timestep) > 500)).all():
            if len(filtered) == 0:
                filtered = pd.DataFrame(base.loc[i]).T
            else:
                filtered = pd.concat([filtered, pd.DataFrame(base.loc[i]).T])

    print "Spacetime filter:", time.time() - time_1
    if len(filtered) == 0:
	print "no valid transients in time interval"
	continue

# Make inspection plot
    
    dbname = AS_db
    ObsDir = "/data/"+dbname+"_Candidates/"



    for _ID in filtered.index: #AART_catsource.index:

        source_df = retrieve_source(transients, _ID)

        _source_flux = source_df.f_int[(source_df.freq_eff < 60000000) ].values
        _index = source_df.extract_type[(source_df.freq_eff < 60000000) ].values
        _source_flux[(_index == 1)] = np.nan


        plt.figure(fig_n, figsize=(12,12))

        stamp = hp.gnomview(map_load_1,  xsize=stamp_side,
                            rot=([base.loc[_ID].ra,
                                  base.loc[_ID].decl]),
                            coord="C", return_projected_map=True,fig=fig_n,sub=333,notext=True, title="Map: 75 MHz",cbar=False)

        hp.projscatter(vlssr[vlssr.Sp > vlssr_thresh]._RAJ2000.values,
                    vlssr[vlssr.Sp > vlssr_thresh]._DEJ2000.values, lonlat=True,
                       marker="+", color="white", coord="C")

        hp.projscatter(base.loc[_ID].ra,
                       base.loc[_ID].decl,lonlat=True,
                       edgecolors="red", facecolor="none", s=1500, coord="C", lw=1)
    #######
        stamp = hp.gnomview(map_load_3,  xsize=stamp_side,
                            rot=([base.loc[_ID].ra,
                                  base.loc[_ID].decl]),
                            coord="C", return_projected_map=True, 
                            fig=fig_n,sub=332, notext=True, title="Map: 60 MHz", cbar=False)

        hp.projscatter(vlssr[vlssr.Sp > vlssr_thresh]._RAJ2000.values,
                    vlssr[vlssr.Sp > vlssr_thresh]._DEJ2000.values, lonlat=True,
                       marker="+", color="white", coord="C")

        hp.projscatter(base.loc[_ID].ra,
                       base.loc[_ID].decl,lonlat=True,
                       edgecolors="red", facecolor="none", s=1500, coord="C", lw=1)

    ########
        stamp = hp.gnomview(map_load_2, xsize=stamp_side,
                            rot=([base.loc[_ID].ra,
                                  base.loc[_ID].decl]),
                            coord="C", return_projected_map=True, fig=1,sub=331, notext=True, title="Map: 37.5 MHz",cbar=False)


        hp.projscatter(vlssr[vlssr.Sp > vlssr_thresh]._RAJ2000.values,
                    vlssr[vlssr.Sp > vlssr_thresh]._DEJ2000.values, lonlat=True,
                       marker="+", color="white", coord="C")

        hp.projscatter(base.loc[_ID].ra,
                       base.loc[_ID].decl,lonlat=True,
                       edgecolors="red", facecolor="none", s=1500, coord="C", lw=1)

        ax = plt.subplot(313)
        
        myFmt = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(myFmt)
        
        plt.errorbar(source_df.taustart_ts[(source_df.freq_eff < 60000000)].values,
                     _source_flux, 
                     yerr = source_df.f_int_err[(source_df.freq_eff < 60000000) ].values,
                     fmt="o-", color="C3",ecolor="C1", label="57.8 MHz, det")

        del _source_flux

        if len(source_df.f_int_err[(source_df.freq_eff < 60000000) & (source_df.extract_type == 1 )].values) > 0:
            plt.errorbar(source_df.taustart_ts[(source_df.freq_eff < 60000000) & (source_df.extract_type == 1 )].values,
                         source_df.f_int[(source_df.freq_eff < 60000000) & (source_df.extract_type == 1 )].values, 
                         yerr = source_df.f_int_err[(source_df.freq_eff < 60000000) & (source_df.extract_type == 1 )].values,
                         fmt=".", color="C3",ecolor="C1", label="57.8 MHz, ff")


        _source_flux = source_df.f_int[(source_df.freq_eff > 60000000) ].values
        _index = source_df.extract_type[(source_df.freq_eff > 60000000) ].values
        _source_flux[(_index == 1)] = np.nan


        plt.errorbar(source_df.taustart_ts[(source_df.freq_eff > 60000000) ].values,
                     _source_flux, 
                     yerr = source_df.f_int_err[(source_df.freq_eff > 60000000) ].values,
                     fmt="o-",color="C0", ecolor="C1", label="61.3 MHz, det")

        del _source_flux

        if len(source_df.f_int_err[(source_df.freq_eff > 60000000) & (source_df.extract_type == 1 )].values) > 0:
            plt.errorbar(source_df.taustart_ts[(source_df.freq_eff > 60000000) & (source_df.extract_type == 1 )].values,
                         source_df.f_int[(source_df.freq_eff > 60000000) & (source_df.extract_type == 1 )].values, 
                         yerr = source_df.f_int_err[(source_df.freq_eff > 60000000) & (source_df.extract_type == 1 )].values,
                         fmt=".", color="C0",ecolor="C1", label="61.3 MHz, ff")


        plt.legend()
        if dbname[:3] == "ASr":
            plt.ylabel("Integrated flux [arbitrary]")
        elif dbname[:3] == "ASf":
            plt.ylabel("Integrated flux [Jy]")
        plt.xlabel("Time [UTC]")

        plt.title("Obs: "+dbname+"\nID: "+str(_ID)+", RA "+str(round(source_df.ra.mean(),2))+", Dec "+str(round(source_df.decl.mean(),2)))

        # if images are on disk make image stamps 
        try:
            if  os.path.isfile(source_df[ source_df.f_int == source_df.f_int.max()].url[0]):
                filename = source_df[ source_df.f_int == source_df.f_int.max()].url[0]
            else: 
                filename = "/mnt/ais001/"+source_df[ source_df.f_int == source_df.f_int.max()].url[0].split("/")[2]+"/"+source_df[ source_df.f_int == source_df.f_int.max()].url[0].split("/")[3]


            wcs = WCS(filename)
            im_pix_x, im_pix_y, n, nn = wcs.wcs_world2pix(source_df.ra.mean(),source_df.decl.mean(),1,1,1)

            pos = [im_pix_x, im_pix_y]
            plt.subplot(3,3,4)
            plt.text(5,92,os.path.basename(filename), color="white", fontsize=12)
            plt.text(0.5,0.5,"max", color="white", fontsize=18)
            plt.imshow(fits.open(filename)[0].data[0,0,pos[1]-delta/2:pos[1]+delta/2,pos[0]-delta/2:pos[0]+delta/2], origin="lower")
            plt.scatter(delta/2,delta/2, s=50*delta, facecolor="none", edgecolor="red")

            plt.xticks([])
            plt.yticks([])

            plt.subplot(3,3,5)
            if  os.path.isfile(source_df[ source_df.f_int == source_df.f_int.min()].url[0]):
                filename = source_df[ source_df.f_int == source_df.f_int.min()].url[0]

            wcs = WCS(filename)
            im_pix_x, im_pix_y, n, nn = wcs.wcs_world2pix(source_df.ra.mean(),source_df.decl.mean(),1,1,1)

            pos = [im_pix_x, im_pix_y]

            plt.text(5,92,os.path.basename(filename), color="white", fontsize=12)
            plt.text(0.5,0.5,"min", color="white", fontsize=18)
            plt.imshow(fits.open(filename)[0].data[0,0,pos[1]-delta/2:pos[1]+delta/2,pos[0]-delta/2:pos[0]+delta/2], origin="lower")
            plt.scatter(delta/2,delta/2, s=50*delta, facecolor="none", edgecolor="red")
            plt.xticks([])
            plt.yticks([])
            
            
        except TypeError:
            print "Image files not on disk"


        ax = plt.subplot(3,3,6)
        plt.text(0.1,0.85,"Max: {}".format(round(source_df[ source_df.f_int == source_df.f_int.max()].f_int.values[0],2)), 
                 color="black",transform=ax.transAxes,fontsize=12)

        plt.text(0.1, 0.80, os.path.basename(source_df[ source_df.f_int == source_df.f_int.max()].url[0]), 
                 color="black",transform=ax.transAxes, fontsize=12)

        plt.text(0.1, 0.68, os.path.basename(source_df[ source_df.f_int == source_df.f_int.min()].url[0]), 
                 color="black",transform=ax.transAxes, fontsize=12)
        plt.text(0.1,0.73,"Min:", color="black",transform=ax.transAxes,fontsize=12)
        plt.plot()
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        
        fig_n +=1
        if not os.path.exists(ObsDir):
            os.makedirs(ObsDir)
        
        if not os.path.exists(ObsDir+"CandidatePandas"):
            os.makedirs(ObsDir+"CandidatePandas")

#         for _ID in filtered.index:
        source_df = retrieve_source(transients, _ID)
        plt.savefig(ObsDir+str(_ID)+"_lightcurve.png")
        source_df.to_csv(ObsDir+"/CandidatePandas/"+str(_ID)+"-source_df.csv", index=False)
	tot_candidate += len(np.unique(transients.runcatid))
	tot_filtered += len(filtered)


print "Total time:", time.time() - tot_timer

myCsvRow = [AS_db,tot_time,tot_candidate, tot_filtered, starttime, endtime ]
with open('/home/kuiack/AS_survey_stats.csv','a') as fd:
	writer = csv.writer(fd)
        writer.writerow(myCsvRow)

sys.exit(0)


