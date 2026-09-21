#!/usr/bin/env python

import os
import sys
import numpy as np

def read_data(fn):
    rmsd = 0.0
    eps = 0.000001
    data = []
    with open(fn) as fp:
        for line in fp:
            if 'CA rmsd=' in line:
                rmsd = float(line.strip().split()[-1])
            elif not line.startswith("#"):
                x = line.strip().split()
                #  N   Nprd  scorecut    |   RMSD    Ave    Med   TP<0.5  TP<1.0  TP<1.5\n
                #x[0]  x[1]     x[2]     ,   x[4]   x[5]   x[6]    x[7]    x[8]   x[9]
                
                n_cryst = int(x[0])
                n_pred  = int(x[1])
                cut     = float(x[2])
                tp_05   = int(x[7])
                tp_10   = int(x[8])
                tp_15   = int(x[9])
        
                if n_cryst == 0:
                    cov_05 = eps
                    cov_10 = eps
                    cov_15 = eps
                else:
                    cov_05  = max(0, tp_05 / n_cryst) 
                    cov_10  = max(0, tp_10 / n_cryst) 
                    cov_15  = max(0, tp_15 / n_cryst) 
                if n_pred == 0:
                    acc_05 = eps
                    acc_10 = eps
                    acc_15 = eps
                else:
                    acc_05  = max(eps, tp_05 / n_pred) 
                    acc_10  = max(eps, tp_10 / n_pred) 
                    acc_15  = max(eps, tp_15 / n_pred) 
        
                f1_05 = 2.0*tp_05/(n_pred + n_cryst)
                f1_10 = 2.0*tp_10/(n_pred + n_cryst)
                f1_15 = 2.0*tp_15/(n_pred + n_cryst)
                ##  N   Nprd   |   RMSD    Ave    Med  pre0.5 pre1.0 pre1.5 rec0.5 rec1.0 rec1.5 f1_0.5 f1_1.0 f1_1.5\n
                data.append([n_cryst, n_pred, cut, x[4],x[5],x[6], acc_05, acc_10, acc_15, cov_05,cov_10, cov_15,f1_05,f1_10,f1_15])
        
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
    sys.stdout.write("#  N   Nprd scorecut |   RMSD    Ave    Med  pre0.5 pre1.0 pre1.5 rec0.5 rec1.0 rec1.5 f1_0.5 f1_1.0 f1_1.5\n")
    #
    for k in range(len(mean_s)):
    #for k in range(6):
        sys.stdout.write("%6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f  %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\n"%tuple(mean_s[k]))
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

