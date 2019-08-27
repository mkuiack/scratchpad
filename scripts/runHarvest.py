#!/usr/bin/python

import multiprocessing as mp
import os
import sys
from subprocess import Popen, PIPE, call 
from joblib import Parallel, delayed


'''Run Multiple parallel streaming candidate harvester processes
ipthyon runHarvester.py n_jobs AS_dbname1 AS_dbname2 ... 
'''


db_list = sys.argv[2:]

def run_harvester(db_name):
	print db_name
	process = call(['ipython', '/home/kuiack/StreamingHarvest.py', db_name ])
	return 

Parallel(n_jobs=int(sys.argv[1]))(delayed(run_harvester)(db_name) for db_name in db_list)

sys.exit()

