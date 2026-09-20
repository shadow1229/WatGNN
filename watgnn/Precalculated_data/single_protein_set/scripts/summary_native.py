#!/usr/bin/env python

import os
import sys
import numpy as np




def read_data(fn):
    rmsd = 0.0
    data = []
    with open(fn) as fp:
        for line in fp:
            if 'CA rmsd=' in line:
                rmsd = float(line.strip().split()[-1])
            elif not line.startswith("#"):
                x = line.strip().split()
                data.append((x[0],x[2],x[3],x[4],x[5],x[6],x[7]))
    data = np.array(data)
    return rmsd, data
    
def run(trglist_fname, result_dir):
    HOME = os.path.dirname(os.path.abspath(__file__))
    targets = [line.strip() for line in open("%s/%s"%(HOME,trglist_fname))]
    
    rmsd_s = []
    data_s = []
    for id in targets:
        fn = '%s/%s/%s.dat'%(HOME,result_dir,id)
        rmsd, data = read_data(fn)
        rmsd_s.append(rmsd)
        data_s.append(data)
    data_s = np.array(data_s, dtype=float)
    mean_s = np.mean(data_s, axis=0)

    sys.stdout.write("# Averaged data\n")
    sys.stdout.write("# Protein CA rmsd= %6.3f\n"%np.mean(rmsd_s))
    sys.stdout.write("#  N |   RMSD    Ave    Med   f<0.5  f<1.0 f<1.5\n")
    #
    for k in range(len(mean_s)):
    #for k in range(6):
        sys.stdout.write("%4d | %6.3f %6.3f %6.3f  %6.4f %6.4f %6.4f\n"%tuple(mean_s[k]))
    #
    sys.stdout.write("#\n")


def main():
    if len(sys.argv) != 3:
        sys.stdout.write("USAGE: %s [trglist_fname] [result_dir]\n"%__file__)
        return 
    else:
        trglist_fname = sys.argv[1]
        result_dir = sys.argv[2]
        sys.stdout.write("# %s\n"%result_dir)
        run(trglist_fname, result_dir)
        sys.stdout.write("#\n")

if __name__=='__main__':
    main()

