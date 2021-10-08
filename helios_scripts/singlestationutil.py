  
"""Functions for working with LOFAR single station data"""

import os
import datetime
from typing import List, Dict, Tuple, Union

import numpy as np
from packaging import version
import tqdm
import h5py

import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.coordinates import SkyCoord, GCRS, EarthLocation, AltAz, get_sun, get_moon
import astropy.units as u
from astropy.time import Time

import lofargeotiff
from lofarantpos.db import LofarAntennaDatabase
import lofarantpos

from .maputil import get_map, make_leaflet_map
from .lofarimaging import nearfield_imager, sky_imager, skycoord_to_lmn, subtract_sources
from .hdf5util import write_hdf5


__all__ = ["sb_from_freq", "freq_from_sb", "find_caltable", "read_caltable",
           "rcus_in_station", "read_acm_cube", "get_station_pqr", "get_station_xyz", "get_station_type",
           "make_sky_plot", "make_ground_plot", "make_xst_plots", "apply_calibration",
           "get_full_station_name", "get_extent_lonlat", "make_sky_movie", "reimage_sky"]

__version__ = "1.5.0"

# Configurations for HBA observations with a single dipole activated per tile.
GENERIC_INT_201512 = [0, 5, 3, 1, 8, 3, 12, 15, 10, 13, 11, 5, 12, 12, 5, 2, 10, 8, 0, 3, 5, 1, 4, 0, 11, 6, 2, 4, 9,
                      14, 15, 3, 7, 5, 13, 15, 5, 6, 5, 12, 15, 7, 1, 1, 14, 9, 4, 9, 3, 9, 3, 13, 7, 14, 7, 14, 2, 8,
                      8, 0, 1, 4, 2, 2, 12, 15, 5, 7, 6, 10, 12, 3, 3, 12, 7, 4, 6, 0, 5, 9, 1, 10, 10, 11, 5, 11, 7, 9,
                      7, 6, 4, 4, 15, 4, 1, 15]
GENERIC_CORE_201512 = [0, 10, 4, 3, 14, 0, 5, 5, 3, 13, 10, 3, 12, 2, 7, 15, 6, 14, 7, 5, 7, 9, 0, 15, 0, 10, 4, 3, 14,
                       0, 5, 5, 3, 13, 10, 3, 12, 2, 7, 15, 6, 14, 7, 5, 7, 9, 0, 15]
GENERIC_REMOTE_201512 = [0, 13, 12, 4, 11, 11, 7, 8, 2, 7, 11, 2, 10, 2, 6, 3, 8, 3, 1, 7, 1, 15, 13, 1, 11, 1, 12, 7,
                         10, 15, 8, 2, 12, 13, 9, 13, 4, 5, 5, 12, 5, 5, 9, 11, 15, 12, 2, 15]

assert version.parse(lofarantpos.__version__) >= version.parse("0.4.0")



def rcus_in_station(station_type):
    """
    Give the number of RCUs in a station, given its type.
    Args:
        station_type: Kind of station that produced the correlation. One of
            'core', 'remote', 'intl'.
    Example:
        >>> rcus_in_station('remote')
        96
    """
    return {'core': 96, 'remote': 96, 'intl': 192}[station_type]


def get_station_type(station_name):
    """
    Get the station type, one of 'intl', 'core' or 'remote'
    Args:
        station_name: Station name, e.g. "DE603LBA" or just "DE603"
    Returns:
        str: station type, one of 'intl', 'core' or 'remote'
    Example:
        >>> get_station_type("DE603")
        'intl'
    """
    if station_name[0] == "C":
        return "core"
    elif station_name[0] == "R" or station_name[:5] == "PL611":
        return "remote"
    else:
        return "intl"


def get_station_pqr(station_name, rcu_mode, db):
    """
    Get PQR coordinates for the relevant subset of antennas in a station.
    Args:
        station_name: Station name, e.g. 'DE603LBA' or 'DE603'
        rcu_mode: RCU mode (0 - 6, can be string)
        db: instance of LofarAntennaDatabase from lofarantpos
    Example:
        >>> from lofarantpos.db import LofarAntennaDatabase
        >>> db = LofarAntennaDatabase()
        >>> pqr = get_station_pqr("DE603", "outer", db)
        >>> pqr.shape
        (96, 3)
        >>> pqr[0, 0]
        1.7434713
        >>> pqr = get_station_pqr("LV614", "5", db)
        >>> pqr.shape
        (96, 3)
    """
    full_station_name = get_full_station_name(station_name, rcu_mode)
    station_type = get_station_type(full_station_name)

    if 'LBA' in station_name or str(rcu_mode) in ('1', '2', '3', '4', 'inner', 'outer', 'sparse_even', 'sparse_odd', 'sparse'):
        if (station_type == 'core' or station_type == 'remote'):
            if str(rcu_mode) in ('3', '4', 'inner'):
                station_pqr = db.antenna_pqr(full_station_name)[0:48, :]
            elif str(rcu_mode) in ('1', '2', 'outer'):
                station_pqr = db.antenna_pqr(full_station_name)[48:, :]
            elif rcu_mode in ('sparse_even', 'sparse'):
                all_pqr = db.antenna_pqr(full_station_name)
                # Indices 0, 49, 2, 51, 4, 53, ...
                station_pqr = np.ravel(np.column_stack((all_pqr[:48:2], all_pqr[49::2]))).reshape(48, 3)
            elif rcu_mode == 'sparse_odd':
                all_pqr = db.antenna_pqr(full_station_name)
                # Indices 1, 48, 3, 50, 5, 52, ...
                station_pqr = np.ravel(np.column_stack((all_pqr[1:48:2], all_pqr[48::2]))).reshape(48, 3)
            else:
                raise RuntimeError("Cannot select subset of LBA antennas for mode " + rcu_mode)
        else:
            station_pqr = db.antenna_pqr(full_station_name)
    elif 'HBA' in station_name or str(rcu_mode) in ('5', '6', '7', '8'):
        selected_dipole_config = {
            'intl': GENERIC_INT_201512, 'remote': GENERIC_REMOTE_201512, 'core': GENERIC_CORE_201512
        }
        selected_dipoles = selected_dipole_config[station_type] + \
            np.arange(len(selected_dipole_config[station_type])) * 16
        station_pqr = db.hba_dipole_pqr(full_station_name)[selected_dipoles]
    else:
        raise RuntimeError("Station name did not contain LBA or HBA, could not load antenna positions")

    return station_pqr.astype('float32')


def get_station_xyz(station_name, rcu_mode, db):
    """
    Get XYZ coordinates for the relevant subset of antennas in a station.
    The XYZ system is defined as the PQR system rotated along the R axis to make
    the Q-axis point towards local north.
    Args:
        station_name: Station name, e.g. 'DE603LBA' or 'DE603'
        rcu_mode: RCU mode (0 - 6, can be string)
        db: instance of LofarAntennaDatabase from lofarantpos
    Returns:
        np.array: Antenna xyz, shape [n_ant, 3]
        np.array: rotation matrix pqr_to_xyz, shape [3, 3]
    Example:
        >>> from lofarantpos.db import LofarAntennaDatabase
        >>> db = LofarAntennaDatabase()
        >>> xyz, _ = get_station_xyz("DE603", "outer", db)
        >>> xyz.shape
        (96, 3)
        >>> f"{xyz[0, 0]:.7f}"
        '2.7033776'
        >>> xyz, _ = get_station_xyz("LV614", "5", db)
        >>> xyz.shape
        (96, 3)
    """
    station_pqr = get_station_pqr(station_name, rcu_mode, db)

    station_name = get_full_station_name(station_name, rcu_mode)

    rotation = db.rotation_from_north(station_name)

    pqr_to_xyz = np.array([[np.cos(-rotation), -np.sin(-rotation), 0],
                           [np.sin(-rotation), np.cos(-rotation), 0],
                           [0, 0, 1]])

    station_xyz = np.matmul((pqr_to_xyz , station_pqr.T).T)

    return station_xyz, pqr_to_xyz



def get_full_station_name(station_name, rcu_mode):
    """
    Get full station name with the field appended, e.g. DE603LBA
    Args:
        station_name (str): Short station name, e.g. 'DE603'
        rcu_mode (Union[str, int]): RCU mode
    Returns:
        str: Full station name, e.g. DE603LBA
    Example:
        >>> get_full_station_name("DE603", '3')
        'DE603LBA'
        >>> get_full_station_name("LV614", 5)
        'LV614HBA'
        >>> get_full_station_name("CS013LBA", 1)
        'CS013LBA'
        >>> get_full_station_name("CS002", 1)
        'CS002LBA'
    """
    if len(station_name) > 5:
        return station_name

    if str(rcu_mode) in ('1', '2', 'outer'):
        station_name += "LBA"
    elif str(rcu_mode) in ('3', '4', 'inner'):
        station_name += "LBA"
    elif 'sparse' in str(rcu_mode):
        station_name += "LBA"
    elif str(rcu_mode) in ('5', '6', '7'):
        station_name += "HBA"
    else:
        raise Exception("Unexpected rcu_mode: ", rcu_mode)

    return station_name


def get_extent_lonlat(extent_m,
                      full_station_name,
                      db=lofarantpos.db.LofarAntennaDatabase):
    """
    Get extent in longintude, latitude
    Args:
        extent_m (List[int]): Extent in metres, in the station frame
        full_station_name (str): Station name (full, so with LBA or HBA)
        db (lofarantpos.db.LofarAntennaDatabase): Antenna database instance
    Returns:
        Tuple[float]: (lon_min, lon_max, lat_min, lat_max)
    """
    rotation = db.rotation_from_north(full_station_name)

    pqr_to_xyz = np.array([[np.cos(-rotation), -np.sin(-rotation), 0],
                           [np.sin(-rotation), np.cos(-rotation), 0],
                           [0, 0, 1]])

    pmin, qmin, _ = np.matmul(pqr_to_xyz.T, (np.array([extent_m[0], extent_m[2], 0])))
    pmax, qmax, _ = np.matmul(pqr_to_xyz.T, (np.array([extent_m[1], extent_m[3], 0])))
    lon_min, lat_min, _ = lofargeotiff.pqr_to_longlatheight([pmin, qmin, 0], full_station_name)
    lon_max, lat_max, _ = lofargeotiff.pqr_to_longlatheight([pmax, qmax, 0], full_station_name)

    return [lon_min, lon_max, lat_min, lat_max]


