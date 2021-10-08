#!/usr/bin/env python
"""Read Lofar AntennaField conf files.
The "centos7" format files are not supported."""
from __future__ import print_function
import sys
import json
def multi_dim(data, shape, typ=float):
    """Reshape 'data' list into a multi dimensional list with the given shape
    conv is a function that is run on each element. Eg. to convert string to float"""
    ret = []
    if len(shape) == 1:
        # We are on the final axis so read and return the required elements
        for i in range(shape[0]):
            ret.append(typ(data[0]))
            del data[0]
    else:
        for i in range(shape[0]):
            ret.append(multi_dim(data, shape[1:], typ))
    return ret
def read_array(stream):
    str_data = stream.readline().rstrip('\n')
    if str_data == "":
        raise ValueError
    while ']' not in str_data:
        str_data = str_data + ' ' + stream.readline().rstrip('\n')
    shape, str_data = str_data.split('[')
    str_data, empty = str_data.split(']')
    assert empty == ''
    str_data = [x for x in str_data.split(' ') if x != '']
    shape = [int(s.strip(' ')) for s in shape.split('x')]
    arr = multi_dim(str_data, shape)
    assert len(str_data) == 0
    return arr
def read_positions(stream):
    positions = [read_array(stream)] # reference position
    try:
        # dipole positions, not provided for HBA "ears" in core station
        positions.append(read_array(stream))
    except ValueError:
        pass
    return positions
def from_file(filename):
    stream = open(filename)
    AntennaField = {"NORMAL_VECTOR": {}, "ROTATION_MATRIX": {}}
    line = stream.readline()
    while line != "":
        while line == "\n" or line.startswith('#'):
            line = stream.readline()
        if "NORMAL_VECTOR" in line or "ROTATION_MATRIX" in line:
            arr_name, band = line.split()
            AntennaField[arr_name][band] = read_array(stream)
        else:
            band = line.rstrip('\n')
            AntennaField[band] = read_positions(stream)
        line = stream.readline()
    return AntennaField
if __name__ == "__main__":
    #dump = json.dumps(from_file('AntennaFields/A12-AntennaField.conf'))
    import numpy as np
    
    #print( np.array(a12['LBA'][0])[0])
    
    a12 = json.loads(json.dumps(from_file('A12-AntennaField.conf')))
    xya12 = np.array(a12['LBA'][1])[:,0]
    xa12=xya12[:,0] + np.array(a12['LBA'][0])[0]
    ya12=xya12[:,1] + np.array(a12['LBA'][0])[1]
    
    a1 = json.loads(json.dumps(from_file('AntennaFields/CS001-AntennaField.conf')))
    xya1 = np.array(a1['LBA'][1])[:,0]
    xa1=xya1[:,0] + np.array(a1['LBA'][0])[0]
    ya1=xya1[:,1] + np.array(a1['LBA'][0])[1]
    
    a2 = json.loads(json.dumps(from_file('AntennaFields/CS002-AntennaField.conf')))
    xya2 = np.array(a2['LBA'][1])[:,0]
    xa2=xya2[:,0] + np.array(a2['LBA'][0])[0]
    ya2=xya2[:,1] + np.array(a2['LBA'][0])[1]
    
    a3 = json.loads(json.dumps(from_file('AntennaFields/CS003-AntennaField.conf')))
    xya3 = np.array(a3['LBA'][1])[:,0]
    xa3=xya3[:,0] + np.array(a3['LBA'][0])[0]
    ya3=xya3[:,1] + np.array(a3['LBA'][0])[1]
    
    a4 = json.loads(json.dumps(from_file('AntennaFields/CS004-AntennaField.conf')))
    xya4 = np.array(a4['LBA'][1])[:,0]
    xa4=xya4[:,0] + np.array(a4['LBA'][0])[0]
    ya4=xya4[:,1] + np.array(a4['LBA'][0])[1]
    
    a5 = json.loads(json.dumps(from_file('AntennaFields/CS005-AntennaField.conf')))
    xya5 = np.array(a5['LBA'][1])[:,0]
    xa5=xya5[:,0] + np.array(a5['LBA'][0])[0]
    ya5=xya5[:,1] + np.array(a5['LBA'][0])[1]
    
    a6 = json.loads(json.dumps(from_file('AntennaFields/CS006-AntennaField.conf')))
    xya6 = np.array(a6['LBA'][1])[:,0]
    xa6=xya6[:,0] + np.array(a6['LBA'][0])[0]
    ya6=xya6[:,1] + np.array(a6['LBA'][0])[1]
    
    a7 = json.loads(json.dumps(from_file('AntennaFields/CS007-AntennaField.conf')))
    xya7 = np.array(a7['LBA'][1])[:,0]
    xa7=xya7[:,0] + np.array(a7['LBA'][0])[0]
    ya7=xya7[:,1] + np.array(a7['LBA'][0])[1]
    
    a11 = json.loads(json.dumps(from_file('AntennaFields/CS011-AntennaField.conf')))
    xya11 = np.array(a11['LBA'][1])[:,0]
    xa11=xya11[:,0] + np.array(a11['LBA'][0])[0]
    ya11=xya11[:,1] + np.array(a11['LBA'][0])[1]
    
    a13 = json.loads(json.dumps(from_file('AntennaFields/CS013-AntennaField.conf')))
    xya13 = np.array(a13['LBA'][1])[:,0]
    xa13=xya13[:,0] + np.array(a13['LBA'][0])[0]
    ya13=xya13[:,1] + np.array(a13['LBA'][0])[1]
    
    a17 = json.loads(json.dumps(from_file('AntennaFields/CS017-AntennaField.conf')))
    xya17 = np.array(a17['LBA'][1])[:,0]
    xa17=xya17[:,0] + np.array(a17['LBA'][0])[0]
    ya17=xya17[:,1] + np.array(a17['LBA'][0])[1]
    
    a21 = json.loads(json.dumps(from_file('AntennaFields/CS021-AntennaField.conf')))
    xya21 = np.array(a21['LBA'][1])[:,0]
    xa21=xya21[:,0] + np.array(a21['LBA'][0])[0]
    ya21=xya21[:,1] + np.array(a21['LBA'][0])[1]
    
    a32 = json.loads(json.dumps(from_file('AntennaFields/CS032-AntennaField.conf')))
    xya32 = np.array(a32['LBA'][1])[:,0]
    xa32=xya32[:,0] + np.array(a32['LBA'][0])[0]
    ya32=xya32[:,1] + np.array(a32['LBA'][0])[1]
    
    import matplotlib.pyplot as plt
    
    print('CS002 ',xa12[0],xa2[48])
    print('CS003 ',xa12[48],xa3[48])
    print('CS004 ',xa12[96],xa4[48])
    print('CS005 ',xa12[144],xa5[48])
    print('CS006 ',xa12[192],xa6[48])
    print('CS007 ',xa12[240],xa7[48])
    print('CS011 ',xa12[288],xa11[48])
    print('CS013 ',xa12[336],xa13[48])
    print('CS001 ',xa12[384],xa1[48])
    print('CS017 ',xa12[432],xa17[48])
    print('CS021 ',xa12[480],xa21[48])
    print('CS032 ',xa12[528],xa32[48])
    
    plt.scatter(xa12,ya12)
    plt.scatter(xa1,ya1,c='r')
    plt.scatter(xa2,ya2,c='g')
    plt.scatter(xa3,ya3,c='c')
    plt.scatter(xa4,ya4,c='y')
    plt.scatter(xa5,ya5,c='w')
    plt.scatter(xa6,ya6,c='cc')
    plt.scatter(xa7,ya7,c='yy')
    plt.scatter(xa11,ya11,c='gg')
    plt.scatter(xa13,ya13,c='rr')
    plt.scatter(xa17,ya17,c='ww')
    plt.scatter(xa21,ya21,c='ww')
    plt.scatter(xa32,ya32,c='ww')
    
    plt.show()
