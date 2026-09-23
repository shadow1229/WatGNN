import random
from scipy.spatial.distance import cdist
from scipy.stats import special_ortho_group #for random rotation
import numpy as np
import torch
import dgl
import os,gc

def partition_pdb_dict(pdb_dict, max_atom=2500, no_partition = 5000): #split pdb_dict into multiple pdb_dict graphs with 

    pos_list = pdb_dict['pos_list']
    n_atm           = pos_list.shape[0] #pos_list: Nx3
    #for debugging
    #print(pos_list_atm.shape[0])
    print('input_preprocess.py - pos_list_shape',pos_list.shape)
    if (pos_list.shape[0] < no_partition) : #less than 5000 atoms
        return [pdb_dict] #no partition
    else:
        n_partitions = max(2, 1+int(pos_list.shape[0]/max_atom))
        print('input_preprocess.py - n_partitions',n_partitions)
        graph_dist_lr = 10.0
        graph_lr =  dgl.radius_graph(pos_list, graph_dist_lr, self_loop=False)
        
        partition_nodes = dgl.metis_partition_assignment(graph_lr, n_partitions)
        partition_masks = [ partition_nodes == k for k in range(n_partitions)]
        #print(partition_masks[1])
        predecessor_list = [ list(set(list(graph_lr.predecessors(k)))) for k in range(pos_list.shape[0]) ]
        #print(predecessor_list[0:4])
        partition_neighbor_masks = [partition_nodes == k for k in range(n_partitions)]
        for k in range(n_partitions):
            for i in range(pos_list.shape[0]):
                if partition_masks[k][i] == True:
                    for n_idx in predecessor_list[i]:
                        partition_neighbor_masks[k][int(n_idx)] = True #n_idx: tensor int / int(n_idx): int

        resno_list_torch   = pdb_dict['resno_list']
        #resno_all_list_torch   = probe_dict['resno_list']
        crop_resno_set          = [list(set(resno_list_torch[partition_masks[k]].tolist())) for k in range(n_partitions)]
        crop_resno_set_neighbor = [list(set(resno_list_torch[partition_neighbor_masks[k]].tolist())) for k in range(n_partitions)]
        crop_mask               = [torch.any(torch.stack([torch.eq(resno_list_torch, crop_resno) for crop_resno in crop_resno_set[k]],dim=0), dim=0) for k in range(n_partitions)]
        crop_mask_neighbor      = [torch.any(torch.stack([torch.eq(resno_list_torch, crop_resno) for crop_resno in crop_resno_set_neighbor[k]],dim=0), dim=0) for k in range(n_partitions)]
        crop_mask_float = [ crop_mask[k].type(torch.float32) for k in range(n_partitions) ]
           

        #print(partitions)
        #remove redundancy
        polar_mask_list_new = [ pdb_dict['polar_mask_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]

        #neighboring N/O atom has 0 in loss_mask_list, but the atom is included in the partitioned graph. 
        loss_mask_list_new =  [torch.einsum('i,i->i',pdb_dict['loss_mask_list'],crop_mask_float[k])[crop_mask_neighbor[k]] for  k in range(n_partitions)]

        #2. build rotation-invariant features
        res_list_new     = [ pdb_dict['res_list'][crop_mask_neighbor[k]] for  k in range(n_partitions)]
    
        resname_list_new = [[] for  k in range(n_partitions)]
    
        atmname_list_new = [[] for  k in range(n_partitions)]
        for k in range(n_partitions):
            for i in range(n_atm):
                if crop_mask_neighbor[k][i] == True:
                    resname_list_new[k].append(pdb_dict['resname_list'][i])
                    atmname_list_new[k].append(pdb_dict['atmname_list'][i])


        resno_list_new   = [pdb_dict['resno_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        atm_list_new     = [pdb_dict['atm_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        charge_list_new =  [pdb_dict['charge_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        hyb_list_new =     [pdb_dict['hyb_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        water_cutoff_new = [pdb_dict['water_cutoff']  for k in range(n_partitions)]
        n_water_list_new = [pdb_dict['n_water_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        n_water_int_list_new =     [pdb_dict['n_water_int_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        n_water_analysis_int_new = [pdb_dict['n_water_analysis_int'][crop_mask_neighbor[k]] for k in range(n_partitions)]

        n_water_ww_list_new =     [pdb_dict['n_water_ww_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        n_water_ww_int_list_new = [pdb_dict['n_water_ww_int_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        n_water_analysis_ww_int_new = [pdb_dict['n_water_analysis_ww_int'][crop_mask_neighbor[k]] for k in range(n_partitions)]

        #bond: should contain both atom index
        atmno_crop_list = [torch.arange(n_atm)[crop_mask_neighbor[k]].tolist() for k in range(n_partitions)]
        atmno_crop_inv =  [ [ None for i in range(n_atm)] for k in range(n_partitions)]
        for k in range(n_partitions):
            for idx, val in enumerate(atmno_crop_list[k]):
                atmno_crop_inv[k][val] = idx

        bond_list = pdb_dict['bond_list']
        bond_list_mask = [torch.all(torch.any(torch.stack([torch.eq(bond_list, crop_atmno) for crop_atmno in atmno_crop_list[k]],dim=0), dim=0), dim=1) for k in range(n_partitions)]
        bond_list_new_tmp = [bond_list[bond_list_mask[k]].tolist() for k in range(n_partitions)]
        #change into new atom index, there seems to be no way faster than this...
        bond_list_new = [torch.tensor([ [atmno_crop_inv[k][old_idx] for old_idx in bond] for bond in bond_list_new_tmp[k] ], dtype=torch.int64) for k in range(n_partitions)]

        pos_list_new = [pdb_dict['pos_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        water_pos_new = pdb_dict['water_pos']

        #4. make protein-water distance matrix
        #To prevent possible loss of far-water ~ protein atom pair, 
        # building distance matrix was done before adding gaussian noise to protein atom positions.
        dist0 = [cdist(pos_list_new[k], water_pos_new) for k in range(n_partitions)]
        for k in range(n_partitions):
            for i in range(polar_mask_list_new[k].shape[0]):
                if polar_mask_list_new[k][i] != 1:
                    dist0[k][i,:] = 999.999
        mindist_new = [[ np.amin(dist0[k][:,j]) for j in range(len(water_pos_new))]for k in range(n_partitions)] #for debugging purpose
    
        #6. polar vector, axis (affected by rotation, gauissian noise) - random rotation only ... for now!
        polar_vec_list_new =   [pdb_dict['polar_vec_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        axis_list_new =        [pdb_dict['axis_list'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        grid_diff_new =        [pdb_dict['grid_diff'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        neigh_water_diff_new = [pdb_dict['neigh_water_diff'][crop_mask_neighbor[k]] for k in range(n_partitions)]
        neigh_water_diff_ww_new = [pdb_dict['neigh_water_diff_ww'][crop_mask_neighbor[k]] for k in range(n_partitions)] 

        result = [{ 
               
                'pos_list':pos_list_new[k],  #position of atoms
                'res_list':res_list_new[k], #residue type embedding
                'resname_list':resname_list_new[k],
                'resno_list':resno_list_new[k], #residue number indexing
                'atm_list':atm_list_new[k], #atom type embedding
                'atmname_list':atmname_list_new[k],                   #atom name
                'bond_list':bond_list_new[k],  #new
                'polar_vec_list':polar_vec_list_new[k], #polar vector new
                'axis_list':axis_list_new[k], #torch tensor.
                'charge_list':charge_list_new[k], #charge from charmm36 new
                'hyb_list':hyb_list_new[k], #hybridization, 0 for sp2, 1 for sp3. new
                'n_water_list':n_water_list_new[k],              #number of water nearby the atom
                'n_water_int_list':n_water_int_list_new[k],
                'n_water_analysis_int':n_water_analysis_int_new[k],
                'grid_diff':grid_diff_new[k], #center of each grid, precalculated for prediction
                'water_pos':water_pos_new,               #position of total water molecules Nx3
                'water_cutoff':water_cutoff_new, #water
                'neigh_water_diff':neigh_water_diff_new[k], # N x max_neigh x 3, saves difference between water crd and protein atom crd.
                'polar_mask_list':polar_mask_list_new[k], #polar atom mask
                'loss_mask_list':loss_mask_list_new[k], #polar atom mask
                'n_water_ww_list':n_water_ww_list_new[k],              #number of water nearby the atom
                'n_water_ww_int_list':n_water_ww_int_list_new[k],
                'n_water_analysis_ww_int':n_water_analysis_ww_int_new[k],
                'neigh_water_diff_ww':neigh_water_diff_ww_new[k], # N x max_neigh x 3, saves difference between water crd and protein atom crd.
                'mindist':mindist_new[k]} for k in range(n_partitions)] 
        del(partition_nodes)
        torch.cuda.empty_cache()
        gc.collect()
        return result

def transform(pdb_dict, config={},crop_pdb_dict=None):
    crop = False
    crop_radius = 999.999
    random_rotation = False
    add_noise = False
    noise_stdev = 0.5
    if 'crop' in config.keys():
        crop = config['crop']
    if 'crop_radius' in config.keys():
        crop_radius = config['crop_radius']
    if 'random_rotation' in config.keys():
        random_rotation = config['random_rotation']

    pos_list = pdb_dict['pos_list']
    #resno_list: N ([0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3....])
    resno_list = pdb_dict['resno_list']

    #print(pos_list.shape)
    #print(resno_list.shape)
    #print(resno_list)
    n_atm           = pos_list.shape[0] #pos_list: Nx3
    
    #1. apply crop 
    crop_mask = None
    if crop == True:
        crop_valid = False
        while (crop_valid == False):
            use_crop_pdb_dict_randval = random.random() 
            if crop_pdb_dict == None or use_crop_pdb_dict_randval < 0.5:
                #for cropping
                center_atom_idx = random.randrange(n_atm)
                #1.1. select position of random atomm, then add gaussian random number with stdev of 3A.
                center_vec = torch.normal(mean=pos_list[center_atom_idx], std=3.0)
            else:
                pos_list_crop = crop_pdb_dict['pos_list']
                n_crop        = pos_list_crop.shape[0]
                if n_crop == 0:
                    center_atom_idx = random.randrange(n_atm)
                    #1.1. select position of random atomm, then add gaussian random number with stdev of 3A.
                    center_vec = torch.normal(mean=pos_list[center_atom_idx], std=3.0)
                else:
                    center_atom_idx = random.randrange(n_crop)
                    #1.1. select position of random atomm, then add gaussian random number with stdev of 3A.
                    center_vec = torch.normal(mean=pos_list_crop[center_atom_idx], std=3.0)

            #1.2. get distance^2 between pos_list and center_vec 
            dist_sq = torch.sum( torch.pow( (pos_list - center_vec) ,2), axis=1)
        
            #1.3. select atoms within crop_radius.
            #crop_mask_dist: crop mask for atom, determined by distance only. 
            crop_mask_dist = (dist_sq < (crop_radius**2)) # shape: N, dtype: boolean  

            ##for polar atom. - remove polar atoms in edge area from training
            crop_loss_mask = (dist_sq < ( max(0,crop_radius-5)**2)) # shape: N, dtype: boolean

            #usage: crop_pos = pos_list[crop_mask] #Nx3 -> Mx3. change in crop_pos will not applied to pos_list

            #1.4. cover every atoms in residue from crop_mask_list (no partial residues)
            crop_resno_set = list(set(resno_list[crop_mask_dist].tolist()))
            #crop_mask: atoms that sharing resno in crop_resno_set, which is set of resno from crop_mask_dist
            #thus, this mask does #4.
            if len(crop_resno_set) == 0:
                continue
            crop_mask = torch.any(torch.stack([torch.eq(resno_list, crop_resno) for crop_resno in crop_resno_set],dim=0), dim=0)
            
            crop_loss_mask_float = crop_loss_mask.type(torch.float32)
            #print(torch.einsum('i,i->i',pdb_dict['polar_mask_list'],crop_polar_mask_float))
            polar_mask_list_new = pdb_dict['polar_mask_list'][crop_mask]
            loss_mask_list_new = torch.einsum('i,i->i',pdb_dict['loss_mask_list'],crop_loss_mask_float)[crop_mask]
            n_polar_atm = float(torch.sum(polar_mask_list_new))

            if n_polar_atm > 0:
                crop_valid = True

    else:
        crop_mask = torch.ones((n_atm,),dtype=torch.bool) #tensor([True,True,....])
        polar_mask_list_new = pdb_dict['polar_mask_list'][crop_mask]

    #2. build rotation-invariant features
    res_list_new     = pdb_dict['res_list'][crop_mask]
    
    resname_list_new = []
    atmname_list_new = []
    for i in range(n_atm):
        if crop_mask[i] == True:
            resname_list_new.append(pdb_dict['resname_list'][i])
            atmname_list_new.append(pdb_dict['atmname_list'][i])

    resno_list_new   = pdb_dict['resno_list'][crop_mask]
    atm_list_new     = pdb_dict['atm_list'][crop_mask]
    charge_list_new = pdb_dict['charge_list'][crop_mask]
    hyb_list_new = pdb_dict['hyb_list'][crop_mask]
    water_cutoff_new = pdb_dict['water_cutoff']
    n_water_list_new = pdb_dict['n_water_list'][crop_mask]
    n_water_int_list_new = pdb_dict['n_water_int_list'][crop_mask]
    n_water_analysis_int_new = pdb_dict['n_water_analysis_int'][crop_mask]

    n_water_ww_list_new = pdb_dict['n_water_ww_list'][crop_mask]
    n_water_ww_int_list_new = pdb_dict['n_water_ww_int_list'][crop_mask]
    n_water_analysis_ww_int_new = pdb_dict['n_water_analysis_ww_int'][crop_mask]

    #bond: should contain both atom index
    atmno_crop_list = torch.arange(n_atm)[crop_mask].tolist()
    atmno_crop_inv = [ None for i in range(n_atm)]
    for idx, val in enumerate(atmno_crop_list):
        atmno_crop_inv[val] = idx

    bond_list = pdb_dict['bond_list']
    bond_list_mask = torch.all(torch.any(torch.stack([torch.eq(bond_list, crop_atmno) for crop_atmno in atmno_crop_list],dim=0), dim=0), dim=1)
    bond_list_new_tmp = bond_list[bond_list_mask].tolist()
    #change into new atom index, there seems to be no way faster than this...
    bond_list_new = torch.tensor([ [atmno_crop_inv[old_idx] for old_idx in bond] for bond in bond_list_new_tmp ], dtype=torch.int64)

    
    #3. apply random rotation
    if random_rotation == True:
        rot = torch.tensor(special_ortho_group.rvs(3),dtype=torch.float32)
        pos_list_new = torch.matmul(pdb_dict['pos_list'][crop_mask],rot) #i jk / kl -> ijl
        #same as pos_list_new_einsum = torch.einsum('ij,jk->ik',pdb_dict['pos_list'][crop_mask],rot)
        water_pos_new = torch.matmul(pdb_dict['water_pos'],rot)

    else:
        pos_list_new = pdb_dict['pos_list'][crop_mask]
        water_pos_new = pdb_dict['water_pos']

    #4. make protein-water distance matrix
    #To prevent possible loss of far-water ~ protein atom pair, 
    # building distance matrix was done before adding gaussian noise to protein atom positions.
    dist0 = cdist(pos_list_new, water_pos_new)
    for i in range(polar_mask_list_new.shape[0]):
        if polar_mask_list_new[i] != 1:
            dist0[i,:] = 999.999
    mindist_new = [ np.amin(dist0[:,j]) for j in range(len(water_pos_new))] #for debugging purpose
    
    #5. polar vector, axis (affected by rotation, gauissian noise) - random rotation only ... for now!
    if random_rotation == True:

        polar_vec_list_new = torch.matmul(pdb_dict['polar_vec_list'][crop_mask],rot)
        axis_list_new = torch.einsum('ijk,kl->ijl',pdb_dict['axis_list'][crop_mask],rot) #to find rotated axis 0
        grid_diff_new = torch.einsum('ijk,kl->ijl',pdb_dict['grid_diff'][crop_mask],rot)
        neigh_water_diff_new = torch.einsum('ijk,kl->ijl',pdb_dict['neigh_water_diff'][crop_mask],rot)   
        neigh_water_diff_ww_new = torch.einsum('ijk,kl->ijl',pdb_dict['neigh_water_diff_ww'][crop_mask],rot)  

    else:
        polar_vec_list_new =   pdb_dict['polar_vec_list'][crop_mask]
        axis_list_new =        pdb_dict['axis_list'][crop_mask]
        grid_diff_new =        pdb_dict['grid_diff'][crop_mask]
        neigh_water_diff_new = pdb_dict['neigh_water_diff'][crop_mask]      
        neigh_water_diff_ww_new = pdb_dict['neigh_water_diff_ww'][crop_mask]      

    result =  { 
               
                'pos_list':pos_list_new,  #position of atoms
                'res_list':res_list_new, #residue type embedding
                'resname_list':resname_list_new,
                'resno_list':resno_list_new, #residue number indexing
                'atm_list':atm_list_new, #atom type embedding
                'atmname_list':atmname_list_new,                   #atom name
                'bond_list':bond_list_new,  #new
                'polar_vec_list':polar_vec_list_new, #polar vector new
                'axis_list':axis_list_new, #torch tensor.
                'charge_list':charge_list_new, #charge from charmm36 new
                'hyb_list':hyb_list_new, #hybridization, 0 for sp2, 1 for sp3. new
                'n_water_list':n_water_list_new,              #number of water nearby the atom
                'n_water_int_list':n_water_int_list_new,
                'n_water_analysis_int':n_water_analysis_int_new,
                'grid_diff':grid_diff_new, #center of each grid, precalculated for prediction
                'water_pos':water_pos_new,               #position of total water molecules Nx3
                'water_cutoff':water_cutoff_new, #water
                'neigh_water_diff':neigh_water_diff_new, # N x max_neigh x 3, saves difference between water crd and protein atom crd.
                'polar_mask_list':polar_mask_list_new, #polar atom mask
                'loss_mask_list':loss_mask_list_new, #polar atom mask
                'n_water_ww_list':n_water_ww_list_new,              #number of water nearby the atom
                'n_water_ww_int_list':n_water_ww_int_list_new,
                'n_water_analysis_ww_int':n_water_analysis_ww_int_new,
                'neigh_water_diff_ww':neigh_water_diff_ww_new, # N x max_neigh x 3, saves difference between water crd and protein atom crd.
                'mindist':mindist_new}
    return result

def merge_pdb_dicts(pdb_dicts): #list of pdb_dict
    data_tmp = {'ent_list':{'is_tensor':False, 'data':[], 'result':None}  } #initial data type
    n_pos_prev_cumul = [0]
    resno_prev_cumul = [0]

    data_tmp['ent_list']['result'] = []
    for pdb_no, pdb_dict in enumerate(pdb_dicts):
        #add ent_list
        n_pos = pdb_dict['pos_list'].shape[0]
        n_pos_cumul = n_pos_prev_cumul[-1] + n_pos
        n_pos_prev_cumul.append(n_pos_cumul)
        
        resno_max = torch.max(pdb_dict['resno_list']) +1 
        resno_cumul = resno_prev_cumul[-1] + resno_max
        resno_prev_cumul.append(resno_cumul)

        ent_list = [pdb_no for i in range(n_pos)]
        data_tmp['ent_list']['result'].extend(ent_list)

        #add else
        for k in pdb_dict.keys():
            if k not in data_tmp.keys():
                is_tensor = torch.is_tensor(pdb_dict[k])
                data_tmp[k] = { 'is_tensor':is_tensor, 'data':[], 'result':None }
            data_tmp[k]['data'].append( pdb_dict[k] )

    #adjust bond_list_value
    for i, dat in enumerate(data_tmp['bond_list']['data']):
        dat += n_pos_prev_cumul[i] #add # of total atoms in previous entities

    #adjust resno
    for i, dat in enumerate(data_tmp['resno_list']['data']):
        dat += resno_prev_cumul[i] #add # of total atoms in previous entities

    for i, dat in enumerate(data_tmp['pos_list']['data']):
        dat += 12000.0*i

    for i, dat in enumerate(data_tmp['water_pos']['data']):
        dat += 12000.0*i

    #water cutoff does not need to stack
    data_tmp['water_cutoff']['result'] = data_tmp['water_cutoff']['data'][0]        
    
    #stack
    for k in data_tmp.keys():
        if data_tmp[k]['result'] != None:
            continue
        if data_tmp[k]['is_tensor']:
            data_tmp[k]['result'] = torch.cat(data_tmp[k]['data'], dim=0)
        else:
            data_tmp[k]['result'] = []
            for d in data_tmp[k]['data']:
                data_tmp[k]['result'].extend(d)
                
    result =  {'merged':True}
    for k in data_tmp.keys():
        result[k] = data_tmp[k]['result']
    return result

def get_probe(pdb_dict):
    res_dict = {"ALA":0 ,"ARG":1 ,"ASN":2 ,"ASP":3 ,
               "CYS":4 ,"GLN":5 ,"GLU":6 ,"GLY":7 ,
               "HIS":8 ,"ILE":9 ,"LEU":10,"LYS":11,
               "MET":12,"PHE":13,"PRO":14,"SER":15,
               "THR":16,"TRP":17,"TYR":18,"VAL":19,
               "HID":8, "HIE":8, "HIP":8,"MSE":12,"PRB":20,"_ELSE":21}
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    polar_atoms = ['N','O',"P","S","F","CL","BR","I",'CA','MG','ZN','HG','MN','CO','FE','NI','CU','CD','CO.OH']


    pos_list = pdb_dict['pos_list']
    loss_mask_list = pdb_dict['loss_mask_list']
    grid_diff = pdb_dict['grid_diff']
    n_grid_pt = grid_diff.shape[1]
    n_atm = len(pos_list)
    n_prb = ( torch.sum(loss_mask_list).to(int).item() )* (n_grid_pt)
    res_prb = res_dict['PRB']
    atm_prb = atm_dict['PRB']
    data_tmp = { } 
    #initial data type: 'key':{ 'is_tensor':is_tensor, 'data':[], 'result':None }
    
    #copy data structure
    for k in pdb_dict.keys():
        if k in ['entities','entity_firstres_list','res_firstatm_list','merged','bond_list','water_pos','water_cutoff','mindist']:
            continue

        if k not in data_tmp.keys():
            is_tensor = torch.is_tensor(pdb_dict[k])
            data_tmp[k] = { 'is_tensor':is_tensor, 'data':None, 'result':None }
            if is_tensor:
                dtype = pdb_dict[k].dtype
                shape = pdb_dict[k].shape

                if len(shape) == 0: #constant
                    data_tmp[k]['data'] = pdb_dict[k]
                    data_tmp[k]['result'] = pdb_dict[k]
                elif k in ['n_water_list','n_water_int_list','n_water_analysis_int','neigh_water_diff','grid_diff', 'n_water_ww_list','n_water_ww_int_list','n_water_analysis_ww_int','neigh_water_diff_ww']:
                    shape_new = [ length for length in shape]
                    shape_new[0] = n_prb
                    shape_new[1] = 1
                    data_tmp[k]['data'] = torch.zeros(shape_new, dtype=dtype)                    
                else:
                    shape_new = [ length for length in shape]
                    shape_new[0] = n_prb
                    data_tmp[k]['data'] = torch.zeros(shape_new, dtype=dtype)
            else:
                data_tmp[k]['data'] = [None for i in range(n_prb)]

    data_tmp['probe_link_list'] = { 'is_tensor':is_tensor, 'data':[], 'result':None }

    #copy data
    loss_mask_cu = [ 0 for i in range(n_atm)]
    for i in range(n_atm-1):
        loss_mask_cu[i+1] = loss_mask_cu[i]+pdb_dict['loss_mask_list'][i].to(int).item() 

    for k in pdb_dict.keys():
        if k in ['entities','entity_firstres_list','res_firstatm_list','merged','bond_list','water_pos','water_cutoff','mindist']:
            continue

        for i in range(n_atm):
            i_prb = loss_mask_cu[i]
            if pdb_dict['loss_mask_list'][i].to(int).item() == 1:
                for j in range(n_grid_pt):
                    new_idx = n_grid_pt*i_prb + j
                    #print(new_idx)
                    if k in ['n_water_list','n_water_int_list','n_water_analysis_int','neigh_water_diff','grid_diff','n_water_ww_list','n_water_ww_int_list','n_water_analysis_ww_int','neigh_water_diff_ww']:
                        data_tmp[k]['data'][new_idx][0] = pdb_dict[k][i][j]
                    else:
                        data_tmp[k]['data'][new_idx] = pdb_dict[k][i]

    #build probe link
    for i in range(n_atm):
        i_prb = loss_mask_cu[i]
        if pdb_dict['loss_mask_list'][i].to(int).item() == 1:
            for j in range(n_grid_pt):
                new_idx = n_grid_pt*i_prb + j            
                data_tmp['probe_link_list']['data'].append([i,n_atm+new_idx]) #for probe link, probe index starts from n_atm
                data_tmp['probe_link_list']['data'].append([n_atm+new_idx,i])        

    #modify data
    for i in range(n_atm):
        i_prb = loss_mask_cu[i]
        if pdb_dict['loss_mask_list'][i] == 1:
            for j in range(n_grid_pt):
                new_idx = n_grid_pt*i_prb + j
                #print(new_idx)
                data_tmp['pos_list']['data'][new_idx] += data_tmp['grid_diff']['data'][new_idx,0]
                data_tmp['neigh_water_diff']['data'][new_idx,0] -= data_tmp['grid_diff']['data'][new_idx,0]  
                data_tmp['neigh_water_diff_ww']['data'][new_idx,0] -= data_tmp['grid_diff']['data'][new_idx,0]  
                data_tmp['res_list']['data'][new_idx] = res_prb
                data_tmp['atm_list']['data'][new_idx] = atm_prb
                data_tmp['resname_list']['data'][new_idx] = "PRB"
                resname_origin = pdb_dict['resname_list'][i]
                atmname_origin = pdb_dict['atmname_list'][i]
                data_tmp['atmname_list']['data'][new_idx] = "%s_%s_%d"%(resname_origin,atmname_origin,j)

 
    result = {'probe_mask_list':torch.zeros( ( n_atm+n_prb,) )}
    result['probe_mask_list'][n_atm:] = 1.0
    result['probe_link_list'] = torch.from_numpy(np.array(data_tmp['probe_link_list']['data'], dtype=np.int64))

    for k in pdb_dict.keys():
        if k in ['entities','entity_firstres_list','res_firstatm_list','merged','water_pos','water_cutoff','mindist']:
            result[k] = pdb_dict[k]
        elif k in ['n_water_list','n_water_int_list','n_water_analysis_int','neigh_water_diff','grid_diff','n_water_ww_list','n_water_ww_int_list','n_water_analysis_ww_int','neigh_water_diff_ww']: #changes shape
            dtype = pdb_dict[k].dtype
            shape = pdb_dict[k].shape
            shape_new = [ length for length in shape]
            shape_new[1] = 1
            atm_data = torch.zeros(shape_new)
            result[k] = torch.cat([atm_data,data_tmp[k]['data']], dim=0)

        elif k == 'bond_list':
            result['bond_list'] = pdb_dict['bond_list']
        else:
            if data_tmp[k]['is_tensor'] == True:
                result[k] = torch.cat([pdb_dict[k],data_tmp[k]['data']],dim=0)
            else:
                result[k] = []
                result[k].extend(pdb_dict[k])
                result[k].extend(data_tmp[k]['data'])
    result['loss_mask_list'] = result['probe_mask_list']
    return result
