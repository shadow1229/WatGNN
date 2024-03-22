#!/usr/bin/env python

import sys
import copy
import numpy as np
from scipy.spatial.distance import cdist
#from Galaxy.utils.supPDB import ls_rmsd

def read_pdb(pdb_fn):
    protein = []
    water = []
    with open(pdb_fn) as fp:
        for line in fp:
            if line.startswith("ATOM"):
                if line[12:16].strip() != 'CA':
                    continue
                protein.append((line[30:38], line[38:46], line[46:54]))
            elif line.startswith("HETATM"):
                if line[17:20] not in ['WAT','HOH']:
                    continue
                if line[12:16].strip() != 'O':
                    continue
                water.append((line[30:38], line[38:46], line[46:54]))
    protein = np.array(protein, dtype=float)
    water   = np.array(water,   dtype=float)
    return protein, water

def map_water(refw, modw, n_pred=[]):
    if len(n_pred) == 0:
        n_pred = [len(modw)]
    #n_water = min(len(refw), len(modw), min(n_pred))
    n_water =len(refw)
    dist0 = cdist(modw, refw)
    #print('dist0_shape',dist0.shape)
    #
    pair_s = []
    for n in n_pred:
        nn = min(n, len(modw))
        dist = copy.deepcopy(dist0)[:nn] #maximum: len(model water)
        #print ('n,dist0[:n]_shape',n, dist.shape)
        pair = []
        #print('n_water',n_water)
        for i in range(n_water):
            k = np.unravel_index(np.argmin(dist), dist.shape) #k: index of minimum dist from dist[:n]
            pair.append(dist[k])
            dist = np.delete(dist, k[0], 0)
            dist = np.delete(dist, k[1], 1)
        #print(len(pair))
        pair_s.append(pair)
    return np.array(pair_s)

def run(ref_fn, mod_fn, verbose=False, n_pred=[]):
    use_model = False
    refp,refw = read_pdb(ref_fn)
    modp,modw = read_pdb(mod_fn)
    #
    #CArmsd,opr = ls_rmsd(modp, refp)
    #modw = np.dot(opr[1], modw.T).T + opr[0]
    #
    n_wat = len(refw)
    if len(n_pred) == 0:
        n_pred = [int(n_wat)*i for i in range(1,50)]
        n_pred.append(len(modw))
    #print(n_pred)
    dist = map_water(refw, modw, n_pred=n_pred)
    #print dist.shape
    rmsd = np.sqrt(np.mean(dist**2, axis=1))
    mdev = np.mean(dist, axis=1)
    medd = np.median(dist, axis=1)
    #
    lt_15= np.where(dist<1.5)[0]  #check
    lt_10= np.where(dist<1.0)[0]  #check
    lt_05= np.where(dist<0.5)[0]
    frac_15 = np.array([len(np.where(lt_15==k)[0])/float(dist.shape[1]) for k in range(dist.shape[0])])
    frac_10 = np.array([len(np.where(lt_10==k)[0])/float(dist.shape[1]) for k in range(dist.shape[0])])
    frac_05 = np.array([len(np.where(lt_05==k)[0])/float(dist.shape[1]) for k in range(dist.shape[0])])
    #
    dev_20 = []
    for dist_i in dist:
        lt_20 = dist_i[np.where(dist_i<2.0)]
        dev_20.append(np.mean(lt_20))
    #
    if verbose:
        sys.stdout.write("# Protein CA rmsd= %6.3f\n"%0.0)
        sys.stdout.write("#  N |   RMSD    Ave    Med   f<0.5  f<1.0  f<1.5\n")
        for k in range(len(n_pred)):
            sys.stdout.write("%4d | %6.3f %6.3f %6.3f  %6.4f %6.4f %6.4f  %6.3f\n"%\
                    (n_pred[k], rmsd[k], mdev[k], medd[k], frac_05[k], frac_10[k], frac_15[k] , dev_20[k]))
    #
    return n_pred, rmsd, mdev, medd, frac_05, frac_10, dev_20

def main():
    if len(sys.argv) < 3:
        sys.stdout.write("USAGE: %s [ref] [prediction]\n"%__file__)
        return 
    for fn in sys.argv[2:]:
        sys.stdout.write("# %s\n"%fn)
        run(sys.argv[1], fn, verbose=True)
        sys.stdout.write("#\n")

def test():
    run("../set/ref/1byi_A.pdb", '../wkgb/native/1byi_A/out.pdb')

if __name__=='__main__':
    main()
    #test()
