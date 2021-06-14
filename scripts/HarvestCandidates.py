
# coding: utf-8

# In[1]:


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
import itertools

import numpy as np
import pandas as pd
import scipy as sp
import datetime
import os

from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

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

# %matplotlib inline

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

# import database username/password/etc.
sys.path.append('/home/kuiack')
from database_info import *
query_loglevel = logging.WARNING  # Set to INFO to see queries, otherwise WARNING


# database = 'raw_16SB'
# database = 'flux_16SB'
# database = 'fluxcal_db'

# databases = ['higher_201702250130','lower_201702250130']


# In[ ]:


## AARTFAAC survey databases:

# survey_stats = pd.DataFrame(
#     [
#     ["_201608311510", 1, 9229,  200, 53], # 9718 MB 
#     ["_201609051647", 1, 10726, 597, 17], # 11 GB 
#     ["_201609070340", 1, 17933, 76,  6],  # 14 GB 
#     ["_201611120632", 1, 28203, 310,  39],# 22 GB  
#     ["_201611132000", 1, 5078, 302,  15], # 4155 MB 
#     ["_201611140501", 1, 13977, 370,  16],# 6170 MB 
#     ["_201702241630", 1, 6488,  43,  2],  # 1734 MB 
#     ["_201702250130", 2, 13073, 68,  1],  # 8713 MB 
#     ["_201702250800", 1, 21426, 0,  0],   # 23 GB 
#     ["_201702251405", 1, 14860, 901,  47],  # 12 GB  , delete 2018 data 
#     ["_201702260116", 1, 10669, 0,  0], # 6463 MB  264.31s
#     ["_201702260800", 1, 21434, 239, 11], # 19 GB  
#     ["_201702261405", 1, 18436, 621,  35], # 9641 MB   491. s
#     ["_201702270350", 1, 4400,  67,  2],   # 975 MB 
#     ["_201702280900", 1, 3512,  23,  0],   # 782 MB
#     ["_201809230412", 1, 17147,  676,  47],   # 12 GB
#     ["_201809280900", 1, 0,  0,  0]   # 35 GB
# #     ["_201809281701", 1, 0,  0,  0],   # 
# #     ["_201809290600", 1, 0,  0,  0]   # 
#     ],
#     columns=["obs","dataset","timestamp","candidate","pass"])


survey_stats = pd.read_csv("/home/kuiack/survey_stats.csv")


# In[ ]:


dbname = sys.argv[1]

ObsDir = "/data/AS"+dbname+"_Candidates/"

t1 = time.time()

if not os.path.exists(ObsDir):
    os.makedirs(ObsDir)

dataset = survey_stats[survey_stats.obs == dbname].dataset.values[0]


print "database name: ",  dbname
logging.getLogger('sqlalchemy.engine').setLevel(query_loglevel)

db = tkp.db.Database(engine=engine, host=host, port=port,
                     user=user, password=password, database=dbname)

db.connect()
session = db.Session()
sources = dump_sources(dbname,dataset, engine, host, port, user, password)
print len(sources)
data = pd.DataFrame(sources)

del sources

data.taustart_ts = pd.to_datetime(data.taustart_ts)

if dbname == "_201702251405":
    bool_index = [x.year == 2018 for x in data.taustart_ts]
    data = data.drop(data.index[bool_index])
    
data = data.drop(data.index[np.abs(data.f_int) > 10e6])

try:
    data["round_times"] = [datetime.datetime.strptime(os.path.basename(x)[:19], 
                                                      "%Y-%m-%dT%H:%M:%S") for x in data.url.values]
except ValueError:
    data["round_times"] = [datetime.datetime.strptime(os.path.basename(x)[:14], 
                                                      "%Y%m%d%H%M%S") for x in data.url.values]

survey_stats.set_value(survey_stats.obs == dbname,"timestamp", len(np.unique(data.round_times)))
survey_stats.to_csv("survey_stats.csv", index=False)

db._configured = False
del db, session

print time.time() - t1, "seconds."


# In[ ]:


N_detections = 0 

reduced = pd.DataFrame([])


t1 = time.time()
for _id in np.unique(data.runcatid):
    if len(data[(data.runcatid == _id) &                 (data.band == 23) &                 (data.extract_type == 0)].set_index("round_times").index.           intersection(data[(data.runcatid == _id) &                              (data.band == 24) &                              (data.extract_type == 0)].set_index("round_times").index )) > N_detections     and (np.max(data[(data.runcatid == _id )].det_sigma) > 8 ):

        if len(reduced) == 0:
            reduced = pd.DataFrame(data[(data.runcatid == _id)])
        else:
            reduced = pd.concat([reduced,data[(data.runcatid == _id)]])

print time.time() - t1


# In[ ]:


base = reduced.groupby("runcatid").median()
base["taustart_ts"] = reduced.groupby("runcatid").first().taustart_ts
base["timestep"] = [x.timestamp() for x in base.taustart_ts]


# In[ ]:


vlssr = pd.read_csv("/home/kuiack/VLSSr.tsv", comment="#", delimiter="\t")
tgss = pd.read_csv("/home/kuiack/TGSSADR1_7sigma_catalog.tsv", delimiter="\t")
aart = pd.read_csv("/home/kuiack/AARTFAAC_catalogue.csv")
ateam = {"ra":[82.88,299.43,350.28,187.07],"decl":[21.98,40.59,58.54,12.66]}



# In[ ]:


aart_coord = SkyCoord(aart.ra.values*u.deg, aart.decl.values*u.deg, frame='fk5')
ateam_coord = SkyCoord(ateam["ra"]*u.deg, ateam["decl"]*u.deg, frame='fk5')

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
                       (np.abs(base.loc[i].timestep - base.drop(index=i).timestep) > 1000)).all():
        if len(filtered) == 0:
            filtered = pd.DataFrame(base.loc[i]).T
        else:
            filtered = pd.concat([filtered, pd.DataFrame(base.loc[i]).T])


# In[ ]:



thresh = 5

_filtered = filtered


pass_list = [] 

plt.figure(figsize=(12,8))

plt.plot(base.ra, 
         base.decl,
         ".", label="{} Candidates".format(len(base)))

plt.plot(AART_catsource.ra, 
         AART_catsource.decl,
         ".", label="{} Catalogued".format(len(AART_catsource)))


plt.plot(_filtered.ra,
         _filtered.decl,
         ".", label="{} Pass Filter".format(len(_filtered)))



plt.scatter(aart.ra,aart.decl,
            edgecolor="C1", facecolor="none",
            s=100, label="AARTFAAC catalogue")


for i in range(len(_filtered)):
    plt.annotate(s=_filtered.index.values[i],
                 xy=(_filtered.ra.values[i],_filtered.decl.values[i]),
                 xycoords="data")
    pass_list.append(_filtered.index.values[i])
    
plt.scatter(ateam["ra"],ateam["decl"],
            edgecolor="black", facecolor="none",
            s=500, label="Ateam")

survey_stats.set_value(survey_stats.obs == dbname,"candidate", len(base))
survey_stats.set_value(survey_stats.obs == dbname,"pass", len(_filtered))
survey_stats.to_csv("survey_stats.csv", index=False)

plt.ylim([0,90])
plt.ylabel("Declination [deg]")
plt.xlabel("Right Ascension [deg]")
plt.legend()
plt.tight_layout()
plt.savefig(ObsDir+dbname+"_skymap.png")


plt.figure(figsize=(12,8))
plt.title(dbname+" scintillation map")
plt.scatter(AART_catsource.ra.astype(float),AART_catsource.decl.astype(float),
            edgecolor="C1", facecolor="none",
            s=1e4*AART_catsource.eta_int.astype(float)/AART_catsource.f_datapoints.astype(float), label="AARTFAAC catalogue")

plt.plot(_filtered.ra,
         _filtered.decl,
         ".", label="{} Pass Filter".format(len(_filtered)))

plt.ylim([0,90])
plt.ylabel("Declination [deg]")
plt.xlabel("Right Ascension [deg]")
plt.legend()
plt.tight_layout()
plt.savefig(ObsDir+dbname+"_scintilationmap.png")


# In[ ]:


for _ID in pass_list:#AART_catsource.index:

    source_df = retrieve_source(data, _ID)


    _source_flux = source_df.f_int[(source_df.freq_eff < 60000000) ].values
    _index = source_df.extract_type[(source_df.freq_eff < 60000000) ].values
    _source_flux[(_index == 1)] = np.nan


    plt.figure(figsize=(10,8))
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
    plt.title(str(_ID)+": ra: "+str(round(source_df.ra.mean(),2))+", dec: "+str(round(source_df.decl.mean(),2)))
    plt.ylabel("Integrated flux [arbitrary]")
    plt.xlabel("Time [UTC]")

    plt.tight_layout()
    plt.savefig(ObsDir+str(_ID)+"_lightcurve.png")
#     plt.show()


