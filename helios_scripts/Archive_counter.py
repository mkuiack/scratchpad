#!/bin/python


import sys
import os
import glob
import numpy as np

CORR_HDR_MAGIC= 0x000000003B98F002 # Magic number in raw corr visibilities.
A06_HDR_MAGIC = 0x4141525446414143
A12_HDR_MAGIC = 0x4141525446414144
LEN_HDR       = 512



class TransitVis (object):
    fname       = None # Name of visibility binary file.
    hdr         = None # record header
    fvis        = None # File pointer to binary file
    nrec        = None # Total number of records in this file
    nbline      = None # Total number of baselines in this file
    tfilestart  = None # First timestamp in the file
    tfileend    = None # Last timestamp in the file

    # Information extracted from an individual header
    magic       = None
    trec        = None # Current records start time
    sub         = None # Subband number
    nant        = None # Total number of dual-pol antennas in the array.
    npol        = None # Number of polarizations in the observation
    pol2rec     = None # Number of polarizations per record.
    nchan       = None # Number of channels in this file
    norigchan   = None # Number of original channels before preprocessing.
    dt          = None # Integration time per record

    # Per polarization information
    flagged_channels= None # Independent per pol.
    flagged_dipoles = None # Independent per pol.
    ateamflux   = None # Independent per pol.
    weights     = None
    nweight     = None

    # Frequency associated with channel or averaged set of channels.
    freq        = None 

    recbuf      = None # Complete, single record read from file
    vis         = None # Usable visibility array per record (both pols)
    tstamps     = None # List containing timestamps of all records in a file.

    deblev      = None # Debug level for messages.
    missrec     = None # Number of records thrown due to mismatched times.

    # Function abstraction for raw and calibrated visibilities
    read_rec    = None # Function object, initialised to the right read func.
    read_hdr    = None # Function object, initialised to the right read func.
    parse_hdr   = None # Function object, initialised to the right parse func.

    def __init__ (self, fname, nant=None, sub=None, nchan=None, npol=None, 
                    arrayconfig=None):
        """ Initialization of internal state corresponding to a visibility dump.
            Since the correlator raw visibility magic number doesn't distinguish
            between AARTFAAC-6 or AARTFAAC-12, and has no subband information, 
            this can be supplied to the init routine.
        """
        # Check if fname is a valid file, and open
        self.fname = fname
        self.fvis = open (self.fname, 'r')
        self.hdr = self.fvis.read (LEN_HDR)
        self.npol = 2;

#         import pdb; pdb.set_trace();
        self.magic, self.tfilestart, trec = struct.unpack ("<Qdd", self.hdr[0:24])

#         if (self.magic == A06_HDR_MAGIC):
        self.nant     = 288 # 
#             self.read_rec = self.read_cal
        self.parse_hdr= self.parse_cal_hdr
        self.read_hdr = self.read_cal_hdr
        self.nchan    = 1 # See record description in parse_cal_hdr
        self.nweight  = (6*7)/2 # Per station weights
        self.pol2rec  = 1



        # New format, stores only upper triangle float ACM
        self.nbline  = self.nant*(self.nant+1)/2
        # 8 due to float complex.
        self.payload = self.nbline * self.nchan * self.pol2rec * 8
        self.recsize = self.payload + LEN_HDR
        fsize = os.path.getsize(fname)

        # NOTE: We need to read 2 recs for getting both polarizations from .cal
        if self.magic == A12_HDR_MAGIC or self.magic == A06_HDR_MAGIC:
            self.nrec = fsize/(self.recsize*2)
        else:
            self.nrec = fsize/(self.recsize) # each record has both pols in .vis
            # First record has a corrupted timestamp in a .vis file
            self.fvis.seek (self.recsize, 0) 
            self.hdr = self.fvis.read (LEN_HDR)
            self.magic, self.tfilestart, trec = struct.unpack ("<Qdd", 
                                                        self.hdr[0:24])

        self.fvis.seek (0)
        print ("Nrecs", self.nrec)
        print ("size", fsize, self.recsize)
        print (self.tfilestart, trec)
        self.dt = datetime.timedelta (seconds=(trec - self.tfilestart))# seconds
        self.tfilestart = datetime.datetime.utcfromtimestamp(self.tfilestart)
        
        # Find the last timestamp in a file.
        self.fvis.seek (0)
        self.fvis.seek ((self.nrec-1) * self.recsize)
        self.hdr = self.fvis.read (LEN_HDR)
        self.magic, self.tfileend, trec, self.sub, self.nant, pol, \
            self.norigchan, _, _, _, _, self.freq = self.parse_hdr (self.hdr)

        self.fvis.seek (0)


    def parse_cal_hdr(self, hdr):
        """
        Parse aartfaac header for calibrated data, independent per pol.
        Two pols are packed one after another, but with independent headers.
        struct output_header_t
        {
          uint64_t magic;                   ///< magic to determine header
          double start_time;                ///< start time (unix)
          double end_time;                  ///< end time (unix)
          int32_t subband;                  ///< lofar subband
          int32_t num_dipoles;              ///< number of dipoles (288 or 576)
          int32_t polarization;             ///< XX=0, YY=1
          int32_t num_channels;             ///< number of channels (<= 64)
          float ateam_flux[5];              ///< Ateam fluxes(CasA, CygA, Tau, Vir, Sun)
          std::bitset<5> ateam;             ///< Ateam active
          std::bitset<64> flagged_channels; ///< bitset of flagged channels (8 byte)
          std::bitset<576> flagged_dipoles; ///< bitset of flagged dipoles (72 byte)
          uint32_t weights[78];             ///< stationweights n*(n+1)/2, n in {6, 12}
          uint8_t pad[48];                  ///< 512 byte block
        };
        Record layout: The incoming channels in the raw visibilities from the
            correlator can be averaged in various combinations. The resulting
            averaged product gets its own record with an independent header.
            The record size for all records is the same, and corresponding data
            products (all records with the same channel integration, e.g.) will
            go into a single file. Other channel groups being averaged will go 
            into separate files.  
        """
        try:
            magic, t0, t1, sub, nant, pol, norigchan = struct.unpack("<Qddiiii",
                                                     hdr[0:40])
            ateamflux = struct.unpack("<fffff", hdr[40:60])
            ateam, flag_chans = struct.unpack("<QQ", hdr[60:76])
            flag_dips = struct.unpack ("<72b", hdr[76:148])
            weights = struct.unpack (("<%di" % self.nweight), 
                                    hdr[148:148+self.nweight*4]) # int weights
        except struct.error:
            print ('### struct unpack error!')
            raise

        assert (magic == A06_HDR_MAGIC or magic == A12_HDR_MAGIC)
        # TODO: Examine the flagged channels to determine the actual frequency 
        # of the averaged channels. Currently return subband center.
        freq = sub * 195312.5 # Hz,
        
        return (magic, datetime.datetime.utcfromtimestamp(t0), datetime.datetime.utcfromtimestamp(t1), 
                sub, nant, pol, norigchan, ateamflux, flag_chans, flag_dips, weights, freq)

    # Return just the header, with metainformation extracted.
    # Faster than doing an actual read of the record payload.
    def read_cal_hdr (self, rec):
        
        assert (self.fvis)
        if (rec != None):
            raise NotImplemented

        # First pol.
        hdr = self.fvis.read (LEN_HDR)
        m, self.trec, t1, s, d, p, c, ateamflux, flag_chan, flag_dip, weights, \
                 freq = self.parse_hdr (hdr)
        self.ateamflux[p] = ateamflux
        self.flagged_dipoles[p] = flag_dip
        self.flagged_channels[p] = flag_chan

        self.fvis.seek (self.payload, 1)

        # Second polarization
        hdr = self.fvis.read (LEN_HDR)
        self.magic, trec, _, sub, self.nant, pol, \
            self.norigchan, ateamflux, flag_chan, flag_dip, weights, \
                    freq = self.parse_hdr (hdr)

        self.fvis.seek (self.payload, 1)
        # Check if both records correspond to the same trec
        # We can choose to keep only the single pol, or reject this record.
        if  (trec != self.trec):
            self.recbuf = 0
            self.fvis.seek (-self.recsize, 1) 
            print ('## Mismatched times: pol %d tim: %s, pol %d tim: %s' \
                    % (p,self.trec, pol, trec))
            self.missrec = self.missrec + 1
            return -1

        self.ateamflux[p] = ateamflux
        self.flagged_dipoles[p] = flag_dip
        self.flagged_channels[p] = flag_chan

        return 1
    
    
def A12_checker(fname):
        LEN_HDR  = 512
        pol2rec  = 1
        nchan    = 3
        nant     = 576
        nbline  = nant*(nant+1)/2
        
        payload = nbline * nchan * pol2rec * 8
        recsize = payload + LEN_HDR
        
        fsize = os.path.getsize(fname)
        
        return fsize/(recsize*2)


tot_seconds = 0

dirs = glob.glob("/zfs/helios/filer?/mkuiack1/202*")
#dirs = glob.glob("/zfs/helios/filer?/mkuiack1/20210[1,2]*")+glob.glob("/zfs/helios/filer?/mkuiack1/202012*")

for obs_dir in dirs: #glob.glob("/zfs/helios/filer0/mkuiack1/*/SB301*"):
#     print fname
    try:
    	fname =  np.random.choice(glob.glob(obs_dir+"/*.vis"))
        tot_seconds += A12_checker(fname)
    except ValueError:
	print(obs_dir)

print(tot_seconds/60./60., "hours")








