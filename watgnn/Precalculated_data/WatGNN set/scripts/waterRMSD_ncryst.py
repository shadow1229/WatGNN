#!/usr/bin/env python

import sys
import copy
import numpy as np
from scipy.spatial.distance import cdist
#from Galaxy.utils.supPDB import ls_rmsd
MAX_DIST = 10.0
def get_npred_score(cuts,scores):
    if scores[0] == None:
        result = [len(scores) for i in range(len(cuts))]
        return result
    else:
        result = [0 for i in range(len(cuts))]
        for npred, score in enumerate(scores):
            for i, cut in enumerate(cuts):
                if score > cut:
                    result[i] = (npred+1) #because npred starts at 0
        return result
def get_npred_ncryst(n_cryst, cutv_list, n_pred):
    result = [min(int(cutv*n_cryst) , n_pred) for cutv in cutv_list]
    return result
def sort_list(vals,idxs):
    zp = zip(idxs,vals)
    z  = [x for idxs, x in sorted(zp)]
    return z        
def read_pdb(pdb_fn,cutoff_min=-90.0,cutoff_max=999.0, sort_water=False):
    protein = []
    water = []
    scorelist_r  = []#reverse
    scorelist    = []#reverse
    with open(pdb_fn) as fp:
        for line in fp:
            if line.startswith("ATOM"):
                if line[17:20] in ['BCD']:
                    try:
                        score = float(line[60:66])
                        if (score <cutoff_max) and (score>cutoff_min):
                            water.append((line[30:38], line[38:46], line[46:54]))
                            scorelist.append(score)
                            scorelist_r.append(-1.0*score)
                    except:
                        #print ("%s - No score"%pdb_fn)
                        water.append((line[30:38], line[38:46], line[46:54]))
                        scorelist.append(None)
                        scorelist_r.append(None)
                    
                elif line[12:16].strip() == 'CA':
                    protein.append((line[30:38], line[38:46], line[46:54]))
                    
            elif line.startswith("HETATM"):
                if line[17:20] not in ['WAT','HOH']:
                    continue
                if line[12:16].strip() != 'O':
                    continue
                try:
                    score = float(line[60:66])
                    if (score <cutoff_max) and (score>cutoff_min):
                        water.append((line[30:38], line[38:46], line[46:54]))
                        scorelist.append(score)
                        scorelist_r.append(-1.0*score)
                except:
                    #print ("%s - No score"%pdb_fn)
                    water.append((line[30:38], line[38:46], line[46:54]))
                    scorelist.append(None)
                    scorelist_r.append(None)
                    
    sort_water_avail = sort_water
    for score in scorelist_r:
        if score == None:
            sort_water_avail = False
            break

    if sort_water_avail:
        water_sorted = sort_list(water,scorelist_r) #to sort with desc order.
        #print (scorelist)
        scorelist.sort()
        scorelist.reverse()
        water   = np.array(water_sorted,   dtype=float)
    else:
        water   = np.array(water,   dtype=float)
    protein = np.array(protein, dtype=float)
    return protein, water, scorelist

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
            if dist.shape[0] == 0:
                pair.append(MAX_DIST)
            else:
                #assign aribitary big distance if n_pred < n_ref
                k = np.unravel_index(np.argmin(dist), dist.shape) #k: index of minimum dist from dist[:n]
                pair.append(dist[k])
                dist = np.delete(dist, k[0], 0)
                dist = np.delete(dist, k[1], 1)
        #print(len(pair))
        pair_s.append(pair)
    return np.array(pair_s)

def run(ref_fn, mod_fn, verbose=False, n_pred=[],label=None):
    use_model = False
    refp,refw,bfacs      = read_pdb(ref_fn)
    modp,modw,mod_scores = read_pdb(mod_fn,sort_water = False)
    #
    if label == 'foldx':
        cuts = [0.0 for i in range(21)]
    elif label == 'rism':
        cuts = [(16.0 - i*1) for i in range(19)]
    elif label == 'wkgb':
        cuts = [(20.0 - i*1) for i in range(21)]
    elif label == 'cnn':
        cuts = [(45.0 - i*2.5) for i in range(19)]
    elif label == 'gnn':
        cuts = [(95.0 - i*5) for i in range(20)]

        
    
    #CArmsd,opr = ls_rmsd(modp, refp)
    #modw = np.dot(opr[1], modw.T).T + opr[0]
    CArmsd= 0
    #
    n_wat = len(refw)
    #print(refw.shape)
    #n_pred = get_npred_score(cuts,mod_scores)
    if len(n_pred) == 0:
        n_pred = [int(n_wat)*i for i in range(1,50)]
        n_pred.append(len(modw))

    #print(n_pred)
    dist = map_water(refw, modw, n_pred=n_pred)
    #print (dist.shape)
    rmsd = np.sqrt(np.mean(dist**2, axis=1))
    mdev = np.mean(dist, axis=1)
    medd = np.median(dist, axis=1)
    #

    lt_15= np.where(dist<1.5)[0]  #check
    lt_10= np.where(dist<1.0)[0]  #check
    lt_05= np.where(dist<0.5)[0]
    #frac_15 = np.array([len(np.where(lt_15==k)[0])/float(dist.shape[1]) for k in range(dist.shape[0])])
    #frac_10 = np.array([len(np.where(lt_10==k)[0])/float(dist.shape[1]) for k in range(dist.shape[0])])
    #frac_05 = np.array([len(np.where(lt_05==k)[0])/float(dist.shape[1]) for k in range(dist.shape[0])])
    TP_15 = np.array([len(np.where(lt_15==k)[0]) for k in range(dist.shape[0])])
    TP_10 = np.array([len(np.where(lt_10==k)[0]) for k in range(dist.shape[0])])
    TP_05 = np.array([len(np.where(lt_05==k)[0]) for k in range(dist.shape[0])])
    #
    dev_20 = []
    for dist_i in dist:
        lt_20 = dist_i[np.where(dist_i<2.0)]
        if lt_20.shape[0] == 0:
            dev_20.append(2.0)
        else:
            dev_20.append(np.mean(lt_20))
    #
    if verbose:
        sys.stdout.write("# Protein CA rmsd= %6.3f\n"%CArmsd)
        sys.stdout.write("#  N   Nprd scorecut |   RMSD    Ave    Med   TP<0.5  TP<1.0  TP<1.5\n")
        for k in range(len(n_pred)):
            sys.stdout.write("%6d %6d %6.3f | %6.3f %6.3f %6.3f  %6d %6d %6d  %6.3f\n"%\
                    (n_wat, n_pred[k], -1, rmsd[k], mdev[k], medd[k], TP_05[k], TP_10[k], TP_15[k] , dev_20[k]))
    #
    return n_pred, rmsd, mdev, medd, TP_05, TP_10, dev_20

def main():
    if len(sys.argv) < 4:
        sys.stdout.write("USAGE: %s [label] [ref] [prediction]\n"%__file__)
        return 
    for fn in sys.argv[3:]:
        sys.stdout.write("# %s\n"%fn)
        run(sys.argv[2], fn, verbose=True, label=sys.argv[1])
        sys.stdout.write("#")

def test():
    run("../set/ref/1byi_A.pdb", '../wkgb/native/1byi_A/out.pdb', label='wkgb')

if __name__=='__main__':
    main()
    #test()
