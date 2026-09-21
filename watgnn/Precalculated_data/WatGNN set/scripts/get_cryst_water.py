import os, glob, time,copy,random
import numpy as np
import pickle
from pathlib import Path

#water: bcut: 40 
# [ same chain with target's chain or 
#   water from other chain and having neighboring target protein atom within 4.5A ] and
# do not have any neighboring ligand atom within 4.5A

#loss mask: becomes 0 when if polar atom has any neighboring water atom that has neighboring ligand atom
# (criteria: both neighboring criteria has 4.5A cutoff) 

def read_targets(fpath):
    result = []
    f = open(fpath,'r')
    lines = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        else:
            #target = line.strip()[:4]
            target = line.strip()
            tsp = target.split(',')
            if len(tsp) <= 1:
                result.append(target)
            else:
                newname = '%s_%s'%(tsp[0],tsp[1])
                result.append(newname)
    f.close()
    return result


def read_and_sort_water(pdbpath,resname_only=None):
    pdb_f = open(pdbpath,'r')
    
    lines = pdb_f.readlines()
    waters = []
    for line in lines:
        lst = line.strip()
        if lst.startswith('ATOM') or lst.startswith('HETATM'):
            if len(lst)< 66:
                continue
            if len(lst)> 66:
                print(len(lst), pdbpath, lst)
            resname = lst[17:20]
            if resname_only != None and (resname not in resname_only):
                continue

            waters.append(lst)

    waters_sort_tmp = sorted( waters, key = lambda line: -1.0*float(line[60:66]) ) #descending order of occupancy -> will be fixed into B-factor column
    
    waters_sort = ''
    for water_i, water_line in enumerate(waters_sort_tmp):
        atmno = water_i + 1
        resno = water_i + 1
        pt = np.array([float(water_line[30+8*i:38+8*i]) for i in range(3)])
        score = float(water_line[60:66])
        txt = 'HETATM%5d  O   HOH X%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno,atmno%10000,*pt, 0.0, score)
        waters_sort += txt
    return waters_sort

def read_water(pdbpath,resname_only=None):
    pdb_f = open(pdbpath,'r')
    
    lines = pdb_f.readlines()
    waters = []
    for line in lines:
        lst = line.strip()
        if lst.startswith('ATOM') or lst.startswith('HETATM'):
            resname = lst[17:20]
            if resname_only != None and (resname not in resname_only):
                continue

            waters.append(lst)

    waters_sort = ''
    for water_i, water_line in enumerate(waters):
        atmno = water_i + 1
        resno = water_i + 1
        pt = np.array([float(water_line[30+8*i:38+8*i]) for i in range(3)])
        txt = 'HETATM%5d  O   HOH X%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno,atmno%10000,*pt, 0.0, 100.0)
        waters_sort += txt
    return waters_sort

#dir_in  = './gnn_wkgb_result_001'
#dir_out = './gnn_wkgb_result_001_sorted'
dir_in  = './gnn_newset_result_001'
dir_out = './gnn_newset_result_001_sorted'
dir_ans = './gnn_newset_result_001_answer'
suffix_in = '_pred.pdb'
suffix_out = '.pdb'
suffix_ans = '.pdb'
Path(dir_out).mkdir(parents=True,exist_ok=True)
Path(dir_ans).mkdir(parents=True,exist_ok=True)
#targets = read_targets('wkgb_targets_foldx.txt')
targets_train = read_targets('train_new_2.txt')
targets_validation = read_targets('validation_new_2.txt')
targets = []
targets.extend(targets_train)
targets.extend(targets_validation)        

resname_only = ['PRD']
resname_only_answer = ['TRU']
"""
for target in targets:
    pdb_in = '%s/%s%s'%(dir_in,target,suffix_in)
    pdb_out = '%s/%s%s'%(dir_out,target,suffix_out)
    if os.access(pdb_out,0):
        continue
    print (target)
    water_pos = read_and_sort_water(pdb_in,resname_only=resname_only)
    pdb_out_f = open(pdb_out,'w')
    pdb_out_f.write(water_pos)
    pdb_out_f.close()
"""
for target in targets:
    pdb_in = '%s/%s%s'%(dir_in,target,suffix_in)
    pdb_out = '%s/%s%s'%(dir_ans,target,suffix_ans)
    if os.access(pdb_out,0):
        continue
    print (target)
    water_pos = read_water(pdb_in,resname_only=resname_only_answer)
    pdb_out_f = open(pdb_out,'w')
    pdb_out_f.write(water_pos)
    pdb_out_f.close()


