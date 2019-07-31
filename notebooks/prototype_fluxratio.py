
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import gridspec
import matplotlib.dates as mdates

import json
import tkp.db
import tkp.config
import logging
import csv
import time
import sys
import itertools
import pylab
import numpy as np
import pandas as pd
import scipy as sp
import healpy as hp
import datetime
import os
import glob
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
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import interpolate, signal


import pymc3 as pm
from scipy.stats import norm

from scipy import linspace
from scipy import pi,sqrt,exp
from scipy.special import erf

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


# In[ ]:


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


# In[1]:


def pdf(x):
    return 1/sqrt(2*pi) * exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/sqrt(2))) / 2

def skewnorm(x,e=0,w=1,a=0):
    t = (x-e) / w
    return 2 / w * pdf(t) * cdf(a*t)


def delta(shape):
    return (shape/pm.math.sqrt(1.0+shape**2.))


def muz(shape):
    return pm.math.sqrt(2./np.pi)*delta(shape) 

def skewness(shape):
    return (4.- np.pi)/2. * ((delta(shape)*pm.math.sqrt(2./np.pi))**3.)/(1.0-(2.0*delta(shape)**2.)/np.pi)**(3./2.)

def sigmaz(shape):
    return np.sqrt(1.-muz(shape)**2.)


def skew_mode(shape):
    return pm.math.sqrt(2.0/np.pi)*delta(shape) -         skewness(shape) * pm.math.sqrt(1.0 - (pm.math.sqrt(2.0/np.pi)*delta(shape))**2 )/2.0 -         (pm.math.sgn(shape) / 2.0) *( pm.math.exp (-(2.0*np.pi)/pm.math.abs_(shape)))


def sk_mode(loc,scale,shape):
    return loc + skew_mode(shape) * scale

def fit_lightcurve(y, draws=500):
    with pm.Model() as model:

        (mu, sigma) = norm.fit(y)

        loc = pm.Normal("loc",mu, 20)
        scale = pm.HalfNormal("scale", sigma)
        skew = pm.Normal("skew", 0, 5)
        mode = pm.Deterministic("mode",sk_mode(loc,scale,skew))
        _y = pm.SkewNormal("y_dist",mu=loc, sd=scale,alpha=skew, observed=y)
        trace = pm.sample(draws=draws)
    mode = pm.summary(trace)[pm.summary(trace).index == "mode"]["mean"].values[0]
    mode_err = pm.summary(trace)[pm.summary(trace).index == "mode"]["sd"].values[0]
        
    return mode, mode_err


# In[ ]:


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
    sources_query = """    SELECT  im.taustart_ts
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


# In[ ]:


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


# In[ ]:


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



# In[ ]:


engine = 'postgresql'
host = 'ads001'
port = 5432
user = 'mkuiack'
password = 'Morecomplicatedpass1234'

query_loglevel = logging.WARNING


# In[ ]:


survey_stats = pd.read_csv("/home/kuiack/survey_stats.csv")


# In[ ]:


dbname = sys.argv[1]
outfile = "/data/AS"+dbname+"_Candidates/AARTFAAC_cat_flux2.csv"
ObsDir = "/data/AS"+dbname+"_Candidates/"

t1 = time.time()

if not os.path.exists(ObsDir):
    os.makedirs(ObsDir)

dataset = survey_stats[survey_stats.obs == dbname].dataset.values[0]
timesteps = survey_stats[survey_stats.obs == dbname].timestamp.values[0]

print "database name: ",  dbname
logging.getLogger('sqlalchemy.engine').setLevel(query_loglevel)

db = tkp.db.Database(engine=engine, host=host, port=port,
                     user=user, password=password, database=dbname)

db.connect()
session = db.Session()
sources = dump_sources(dbname, dataset, engine, host, port, user, password)
print len(sources)
data = pd.DataFrame(sources)

del sources

data.taustart_ts = pd.to_datetime(data.taustart_ts)

# Remove bad source fits 
data = data.drop(data.index[np.abs(data.f_int) > 10e6])
# No observations are greater than 24 hours 
data = data[data.taustart_ts.diff() < datetime.timedelta(seconds=24*3600)]

# try:
#     data["round_times"] = [datetime.datetime.strptime(os.path.basename(x)[:19], 
#                                                       "%Y-%m-%dT%H:%M:%S") for x in data.url.values]
# except ValueError:
#     data["round_times"] = [datetime.datetime.strptime(os.path.basename(x)[:14], 
#                                                       "%Y%m%d%H%M%S") for x in data.url.values]

# survey_stats.set_value(survey_stats.obs == dbname,"timestamp", len(np.unique(data.round_times)))
# survey_stats.to_csv("survey_stats.csv", index=False)

db._configured = False
del db, session

print time.time() - t1, "seconds."


# In[ ]:


# Detection in both subbands, simultaneously, in > N_detections timesteps. 

# N_detections = 0 


# # multi_detections = [] 
# reduced = pd.DataFrame([])


# t1 = time.time()
# for _id in np.unique(data.runcatid):
#     if len(data[(data.runcatid == _id) & \
#                 (data.band == 23) & \
#                 (data.extract_type == 0)].set_index("round_times").index.\
#            intersection(data[(data.runcatid == _id) & \
#                              (data.band == 24) & \
#                              (data.extract_type == 0)].set_index("round_times").index )) > N_detections \
#     and (np.max(data[(data.runcatid == _id )].det_sigma) > 8 ):

#         if len(reduced) == 0:
#             reduced = pd.DataFrame(data[(data.runcatid == _id)])
#         else:
#             reduced = pd.concat([reduced,data[(data.runcatid == _id)]])

# print time.time() - t1


# In[ ]:


base = data.groupby("runcatid").median()
# base["taustart_ts"] = data.groupby("runcatid").first().taustart_ts
base["f_datapoints"] = data.groupby("runcatid").last().f_datapoints
# base["timestep"] = [x.timestamp() for x in base.taustart_ts]


# In[ ]:


# vlssr = pd.read_csv("/home/kuiack/VLSSr_gt_5.csv", comment="#")
# tgss = pd.read_csv("/home/kuiack/TGSSADR1_7sigma_catalog.tsv", delimiter="\t")
aart = pd.read_csv("/home/kuiack/AARTFAAC_catalogue.csv")
# ateam = {"ra":np.array([82.88,299.43,350.28,187.07]),
#          "decl":np.array([21.98,40.59,58.54,12.66])}


# In[ ]:


aart_coord = SkyCoord(aart.ra.values*u.deg, aart.decl.values*u.deg, frame='fk5')
# ateam_coord = SkyCoord(ateam["ra"]*u.deg, ateam["decl"]*u.deg, frame='fk5')
AART_catsource = pd.DataFrame([], columns=base.keys())

for i in base[base.f_datapoints > 1800].index:
    try:
        c1 = SkyCoord(base.loc[i].ra*u.deg, base.loc[i].decl*u.deg, frame='fk5')

        c2 = SkyCoord(base.drop(index=i).ra.values*u.deg, 
                  base.drop(index=i).decl.values*u.deg, frame='fk5')
    except IndexError:
        print i

    if np.min(c1.separation(aart_coord).deg) < 1:
        if len(AART_catsource) == 0:
            AART_catsource = pd.DataFrame(base.loc[i]).T
        else:
            AART_catsource = pd.concat([AART_catsource, pd.DataFrame(base.loc[i]).T])


# In[ ]:


if not os.path.exists(outfile):
    fields=['ra','decl','mode_lo','mode_lo_err','mode_hi','mode_hi_err','N']
    with open(outfile, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

for _ID in AART_catsource.index:

    mode_lo,mode_lo_err = fit_lightcurve(clip(data[(data.band == 23) &                                               (data.runcatid == _ID) &                                               (data.extract_type == 0)].f_int.values), draws=100)

    mode_hi,mode_hi_err = fit_lightcurve(clip(data[(data.band == 24) &                                               (data.runcatid == _ID) &                                               (data.extract_type == 0)].f_int.values), draws=100)

    N = len(data[(data.band == 24) & (data.runcatid == _ID)].f_int.values)
    ra = np.nanmean(data[(data.band == 24) & (data.runcatid == _ID)].ra.values)
    decl = np.nanmean(data[(data.band == 24) & (data.runcatid == _ID)].decl.values)

    with open(outfile, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([ra,decl,mode_lo,mode_lo_err,mode_hi,mode_hi_err,N])

