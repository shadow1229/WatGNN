import psutil
import os, time,copy,gc
import numpy as np

import torch
import pickle
import dgl
from scipy.spatial.distance import cdist

from watgnn_input_preprocess import transform, merge_pdb_dicts, get_probe, partition_pdb_dict
from watgnn_input import read_paths,read_pdb_old,read_dataset, read_dataset_simple


def mem():
    my_mem = psutil.virtual_memory()
    return 'Memory used: {:.2f} %, {:.2f} MB | free: {:.2f} MB'.format(my_mem.percent,my_mem.used/1024/1024,my_mem.free/1024/1024)

EPS = 0.00001
MAX_DIST = 10.0
def map_water(refw, modw):
    n_water =len(refw)
    if n_water == 0:
        return np.array([])
    dist0 = cdist(modw, refw)
    pair_s = []
    dist = copy.deepcopy(dist0)
    #print('n_water',n_water)
    for i in range(n_water):
        if dist.shape[0] == 0:
            pair_s.append(MAX_DIST)
        else:
            k = np.unravel_index(np.argmin(dist), dist.shape) #k: index of minimum dist from dist[:n]
            pair_s.append(min(MAX_DIST,dist[k]))
            dist = np.delete(dist, k[0], 0)
            dist = np.delete(dist, k[1], 1)
    return np.array(pair_s)

#better analysis for pred_n wise / axis wise
#better filtering (sort edge with max(score) ) DONE - nov very powerful ~ 2%p increase
def eval_dataset(model, dataset, dataset_lig, config, log_dir = 'gnn_log', log_path = 'gnn_eval_log.txt', result_pdb_dir = 'gnn_result',label='None' ):
    if not os.access(result_pdb_dir,0):
        os.mkdir(result_pdb_dir)
    grid_start = config['grid_start']  #start point of grid (-4.5A from atom crd)
    interval   = config['interval']     #grid interval
    n_grid     = config['n_grid']         #maximum number of grid
    water_cutoff = config['water_cutoff']
    debug = config['debug']    
    score_cutoff = config['score_cutoff']   
    performances = {'all':{}, 'filt':{}, 'clust':{}}
    if not os.access(log_dir,0):
         os.mkdir(log_dir)
    logf = open('%s/%s'%(log_dir,log_path),'w')
    timef = open('%s/%s_%s'%(log_dir,log_path,'time.log'),'w')
    errf = open('%s/error.log'%(log_dir),'a')
    model.eval()

    is_ligand = False
    if dataset_lig == None:
        is_ligand = False
    else:
        is_ligand = True
    debug_global_data = {}
    with torch.no_grad():
        for trgidx, pdbpath_chain in enumerate(dataset):
            
            pdb_name = pdbpath_chain[0].split('/')[-1].split('.')[0] 
            if pdbpath_chain[1] == None:
                outf_name = '%s/%s_pred.pdb'%(result_pdb_dir,pdb_name)
            else:
                outf_name = '%s/%s_%s_pred.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1])
            if os.access(outf_name,0):
                continue
                
            if dataset_lig[trgidx] == None:
                is_ligand = None
            
            read_time_start = time.time()
            if is_ligand:
                pdb_dict = read_paths([pdbpath_chain,dataset_lig[trgidx]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 

            else:
                pdb_dict = read_paths([pdbpath_chain],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 

            read_time_end = time.time()    
            read_time = read_time_end - read_time_start               
            item = {'path':pdbpath_chain[0],
                       'pdb_dict':pdb_dict, 
                       'label':label, 
                       'read_time':read_time, 
                       'gnn_time':None,
                       'pos_time':None,
                       'pos_all':None,
                       'pos_filt':None, 
                       'pos_clust':None}

            pdb_dict = item['pdb_dict']
            pdb_path = item['path']
            pdb_name = pdb_path.split('/')[-1].split('.')[0] 
            
            gnn_time_start = time.time()
            print('%s %s'%(pdbpath_chain[0],pdbpath_chain[1]))

            pdb_dict_partitioned = partition_pdb_dict(pdb_dict, max_atom=2500, no_partition = 5000)
            n_partitions = len(pdb_dict_partitioned)
            
            pred_list = [None for k in range(n_partitions)]
            probe_dict = [ get_probe(pdb_dict) for pdb_dict in pdb_dict_partitioned]

            for k, pdb_dict in enumerate(pdb_dict_partitioned):

                n_try = 0
                max_n_try=10
                done = False
                #due to VRAM allocation problem - cannot allocate memory
                while (done == False and n_try < max_n_try):
                    try:
                        pred_list[k], loss,metric = model.forward(pdb_dict, pdb_path=pdb_path, chain=pdbpath_chain[1], save_input=debug)) # currently, metric = (loss_0, loss_1)
                        done = True
                    except Exception as e:
                        n_try += 1
                        print('allocation error : %s %s'%(pdbpath_chain[0],pdbpath_chain[1]))
                        errf.write('eval_dataset - allocation error : %s %s\n'%(pdbpath_chain[0],pdbpath_chain[1]))
                        errf.write('Error log: %s\n\n'%e)
                        torch.cuda.empty_cache()
                        time.sleep(5) #pause 5 seconds  
                
                if n_try >= max_n_try:
                    raise ValueError             
                
                #try:               
                #    pred_list[k], loss,metric = model.forward(pdb_dict, pdb_path=pdb_path, chain=pdbpath_chain[1], save_input=debug) # currently, metric = (loss_0, loss_1)
                #except Exception as e:
                #    print('allocation error : %s %s'%(pdbpath_chain[0],pdbpath_chain[1]))
                #    errf.write('eval_dataset - allocation error : %s %s\n'%(pdbpath_chain[0],pdbpath_chain[1]))
                #    errf.write('Error log: %s\n\n'%e)
                #    continue
                print('partition %d / %d'%(k+1, n_partitions), mem())
            gnn_time_end = time.time()
            gnn_time = gnn_time_end - gnn_time_start
            item['gnn_time'] = gnn_time                        
            
            pos_time_start = time.time()

            pred_diff = torch.cat([ pred_list[k]['pred_vecs'] for k in range(n_partitions)], 0)
            n_waters_pred = torch.cat([ pred_list[k]['n_water_pred'] for k in range(n_partitions)], 0)
            # for debugging purpose / None if config['debug'] = False,
            #data_graph_dist = [ pred_list[k]['data_graph_dist'] for k in range(n_partitions)]
            

            #pred_diff: N x MAX_WAT
            n_waters_pred_torch = n_waters_pred.detach().cpu() 
            pred_diff_detach = pred_diff.detach().cpu()
            pos_list_detach = torch.cat([ probe_dict[k]['pos_list'].detach() for k in range(n_partitions)], 0)#already on cpu

            #check existence of input atoms.
            is_valid = True
            if len(pos_list_detach.shape) == 0 or pos_list_detach.shape[0] == 0:
                print ('ERR empty pos_list_np - %s'%(pdb_path))
                errf.write('ERR empty pos_list_np - %s\n'%(pdb_path))
                continue
            #following code is equivalent to commented code:
            #pred_np = pred_diff_detach.numpy()
            #n_atm = pos_list_np.shape[0]
            #for i in range(n_atm):
            #    for j in range(pred_np.shape[1]):
            #        pred_np[i][j] += pos_list_np[i]
            n_out_channels = pred_diff_detach.shape[1]
            pos_list_repeat = pos_list_detach.unsqueeze(1).repeat(1,n_out_channels,1)
            print('n_waters_pred_torch', n_waters_pred_torch.shape, 'pred_diff_detach', pred_diff_detach.shape, 'pos_list_detach', pos_list_detach.shape, 'pos_list_repeat', pos_list_repeat.shape) 
            pred_torch = (pred_diff_detach + pos_list_repeat) 
            pred_np = pred_torch.numpy() 

            if len(pred_np.shape) < 1:
                print ('ERR empty pred_np - %s'%(pdb_path))
                errf.write('ERR empty pred_np - %s\n'%(pdb_path))
                continue
            #ignore prediction on non-probe atom.
            probe_mask = torch.cat([probe_dict[k]['probe_mask_list'].detach().cpu() for k in range(n_partitions)], 0)
            probe_mask_bool = probe_mask > 0  

            pred_all_torch = pred_torch[probe_mask_bool] 
            pred_n_all_torch = n_waters_pred_torch[probe_mask_bool] 

            pred_all_pw_torch   = pred_all_torch[:,0,:] 
            pred_n_all_pw_torch = pred_n_all_torch[:,0] 
            
            pred_all_ww_torch = pred_all_torch[:,1,:] 
            pred_n_all_ww_torch = pred_n_all_torch[:,1]  
            print('pred_all', mem())
            #filtering with score
            pw_score_mask = (pred_n_all_pw_torch > score_cutoff) 
            ww_score_mask = (pred_n_all_ww_torch > score_cutoff)

            pred_filt_pw_torch = pred_all_pw_torch[pw_score_mask] 
            pred_n_filt_pw_torch = pred_n_all_pw_torch[pw_score_mask] 
            pred_filt_pw = pred_filt_pw_torch.numpy() 
            pred_n_filt_pw = pred_n_filt_pw_torch.numpy() 
            pred_types_pw = np.zeros_like(pred_n_filt_pw) 

            pred_filt_ww_torch = pred_all_ww_torch[ww_score_mask] 
            pred_n_filt_ww_torch = pred_n_all_ww_torch[ww_score_mask] 
            pred_filt_ww = pred_filt_ww_torch.numpy()
            pred_n_filt_ww = pred_n_filt_ww_torch.numpy() 
            pred_types_ww = np.zeros_like(pred_n_filt_ww) 

            pred_filt_tmp = np.concatenate((pred_filt_pw,pred_filt_ww), axis = 0)
            pred_n_filt_tmp = np.concatenate((pred_n_filt_pw,pred_n_filt_ww), axis = 0)
            pred_types = list(np.concatenate((pred_types_pw, pred_types_ww) , axis = 0))
            print('pred_filt', mem())

            #sort pred_filt, pred_n_filt with pred_n_filt
            pred_filt_pos_n = zip(pred_filt_tmp,pred_n_filt_tmp)

            if len(pred_filt_tmp.shape) < 1 or pred_filt_tmp.shape[0] < 1:
                print ('ERR empty pred_filt - %s'%pdb_path)
                errf.write('ERR empty pred_filt - %s\n'%pdb_path)
                pred_filt = []
                pred_n_filt = []
                pt_clust = []
                n_clust = []

            else:
                pred_filt_pos_n_sort = sorted(pred_filt_pos_n, key = lambda t: -1.0*t[1])
                pred_filt, pred_n_filt = zip(*pred_filt_pos_n_sort)
                print('pred_sort', mem())

                #clustering
                pred_filt_torch = torch.from_numpy(np.array(pred_filt))
                n_max_water = min(pred_filt_torch.shape[0], 30000) 
                print('n_max_water: ',pred_filt_torch.shape[0], 30000)
                clust_indice = [ i for i in range(len(pred_filt))]
                removed_indice = []

                try:
                    #pred_types = [] #0:p-w / 1:w-w
                    pred_graph =  dgl.radius_graph(pred_filt_torch[:n_max_water], config['clust_radius'], self_loop=False) #check collision between predicted water positions
                    #dgl.radius_graph: bidirected graph
                    edge_src, edge_dst = pred_graph.edges(order='srcdst') #shape: n_edge, order: src id - dst id, vertices are sorted with pred_n, and the graph is bidirectional
                    #thus, ordering with srcdst will be same as sorting with max(pred_n_filt[src],pred_n_filt[dst])
                    edge_src = edge_src.detach().cpu().numpy()
                    edge_dst = edge_dst.detach().cpu().numpy()
                    print('pred_edge', mem())
                    for idx in range(edge_src.shape[0]):            
                        src = edge_src[idx]
                        dst = edge_dst[idx]

                        #ignore pair contains removed point
                        if src in removed_indice or dst in removed_indice:
                            continue
                        #remove point with smaller pred_n, which could be probability-like score for position
                        n_src = pred_n_filt[src]
                        n_dst = pred_n_filt[dst]
                        #old method:
                        if n_src < n_dst:
                            removed_indice.append(src)
                        else:
                            removed_indice.append(dst)
                        #this one is actually channel index of the result.
                        # 0: p-w: higher priority / 1: w-w: lower priority
                        """
                        type_src = pred_types[src]
                        type_dst = pred_types[dst]
                        if type_src < type_dst:
                            removed_indice.append(dst)

                        elif type_dst < type_src:
                            removed_indice.append(src)
                    
                        else:
                            if n_src < n_dst:
                                removed_indice.append(src)
                            else:
                                removed_indice.append(dst)
                        """
                    pt_clust = []
                    n_clust  = []
                    for idx in clust_indice:
                        if idx in removed_indice:
                            continue
                        pt = pred_filt[idx]
                        pn = pred_n_filt[idx]
                        pt_clust.append(pt)
                        n_clust.append(pn)
                    

                except:
                    print ('ERR clustering- %s'%pdb_path)
                    errf.write('ERR clustering - %s\n'%pdb_path)
                    pt_clust = []
                    n_clust  = []
            print('pred_clust', mem())
            #for analysis
            pred_all_pw = pred_all_pw_torch.numpy()
            pred_n_all_pw = list(pred_n_all_pw_torch.numpy())
            pred_all_ww = pred_all_ww_torch.numpy()
            pred_n_all_ww = list(pred_n_all_ww_torch.numpy())
            pred_all = np.concatenate((pred_all_pw,pred_all_ww), axis = 0)
            pred_n_all = np.concatenate((pred_n_all_pw,pred_n_all_ww), axis = 0)

            n_waters_pred_np = n_waters_pred_torch.numpy()

            pos_list_np     = torch.cat( [probe_dict[k]['pos_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
            water_pos_np    = torch.cat( [probe_dict[k]['water_pos'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
            probe_mask_list = torch.cat( [probe_dict[k]['probe_mask_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()

            neigh_water_diff_np    = torch.cat( [probe_dict[k]['neigh_water_diff'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
            neigh_water_diff_ww_np = torch.cat( [probe_dict[k]['neigh_water_diff_ww'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
            water_pos_pw_np = neigh_water_diff_np[...,0,:] + pos_list_np #N x 1 x 3 -> N x 3
            water_pos_ww_np = neigh_water_diff_ww_np[...,0,:] + pos_list_np
            n_water_list_np    = torch.cat( [probe_dict[k]['n_water_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
            n_water_ww_list_np = torch.cat( [probe_dict[k]['n_water_ww_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
            item['pos_all'] = pred_all
            item['pos_filt'] = pred_filt            
            item['pos_clust'] = pt_clust

            
            #atom positions (related with probe position)
            if debug:
                if 'positions' not in debug_global_data.keys():
                    debug_global_data['positions'] = {}
                pos_list_np   = torch.cat( [probe_dict[k]['pos_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
                resno_list_np = torch.cat( [probe_dict[k]['resno_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
                #print(probe_dict['resno_list_str'])
                resname_list = []
                atmname_list = []
                resno_list_str = []
                for k in range(n_partitions):
                    resname_list.extend(probe_dict[k]['resname_list'])
                    atmname_list.extend(probe_dict[k]['atmname_list'])
                    resno_list_str.extend(probe_dict[k]['resno_list_str'])
                axis_list_np = torch.cat( [probe_dict[k]['axis_list'].detach().cpu() for k in range(n_partitions)],0 ).numpy()
                #print(n_water_list_np.shape, n_water_ww_list_np.shape, neigh_water_diff_np.shape, neigh_water_diff_ww_np.shape, n_waters_pred_np.shape, pred_np.shape)

                resno_set = {}
                for i_atm in range(pos_list_np.shape[0]):
                    resno = resno_list_np[i_atm]
                    if resno not in resno_set.keys():
                        resno_set[resno] = {'atm':[],'prb':[],'resno':resno_list_str[i_atm]}
                
                    if resno_list_str[i_atm] != resno_set[resno]['resno']:
                        print('ERROR', resno_set[resno]['resno'], resno_list_str[i_atm])
                    resname = resname_list[i_atm]
                    if resname.startswith("PRB"):
                        resno_set[resno]['prb'].append(i_atm)
                    else:
                        resno_set[resno]['atm'].append(i_atm)
                    #atmname = atmname_list[i_atm]
                    #axis = axis_list_np[i_atm]

            
                    #print(resno, resname, atmname, axis)

                resno_keys = sorted(resno_set.keys())
                for res in resno_keys:
                    for prb_idx in resno_set[res]['prb']:
                        prb_name = atmname_list[prb_idx] #(resname)_(atmname)_(probe index)
                        prb_pos_tmp = pos_list_np[prb_idx]
                        prb_axis = axis_list_np[prb_idx] #[ [ax0] [ax1] [ax2]]
                        prb_pos = np.einsum('ij,j->i',prb_axis, prb_pos_tmp) #rotated to prb_axis coordinate
                        if prb_name not in debug_global_data['positions'].keys():
                            debug_global_data['positions'][prb_name] = {'prev_c_pos':[],'atmname':[], 'atmpos':[],'water_pw':[],'water_ww':[],'pred_pw':[], 'pred_ww':[], 'prb':[], 'axis_rot':[], 'pdbname':[],'resno':[],'resno_str':[]}
                        debug_global_data['positions'][prb_name]['atmname'].append([])
                        debug_global_data['positions'][prb_name]['atmpos'].append([])
                        #debug_global_data['positions'][prb_name]['atmpos_orig'].append([])
                        debug_global_data['positions'][prb_name]['pdbname'].append([])
                        debug_global_data['positions'][prb_name]['water_pw'].append([])
                        debug_global_data['positions'][prb_name]['water_ww'].append([])
                        debug_global_data['positions'][prb_name]['pred_pw'].append([])
                        debug_global_data['positions'][prb_name]['pred_ww'].append([])
                        debug_global_data['positions'][prb_name]['prb'].append(prb_pos_tmp)
                        debug_global_data['positions'][prb_name]['resno'].append(res)
                        debug_global_data['positions'][prb_name]['resno_str'].append(resno_set[res]['resno'])
                        #debug_global_data['positions'][prb_name]['axis_orig'].append(prb_axis)


                        axis_rot = np.einsum('ij,kj->ki',prb_axis, prb_axis) #rotated to prb_axis coordinate           
                        debug_global_data['positions'][prb_name]['axis_rot'].append(axis_rot)               
                        #axis_rot0 = np.einsum('ij,j->i',prb_axis, prb_axis[0]) #rotated to prb_axis coordinate      
                        #axis_rot1 = np.einsum('ij,j->i',prb_axis, prb_axis[1]) #rotated to prb_axis coordinate      
                        #axis_rot2 = np.einsum('ij,j->i',prb_axis, prb_axis[2]) #rotated to prb_axis coordinate
                        if res == 0:
                            debug_global_data['positions'][prb_name]['prev_c_pos'].append(None)
                        else:
                            prev_c_pos_tmp = None
                            found_pos = False
                            for atm_idx in resno_set[res-1]['atm']:
                                atmname = atmname_list[atm_idx]
                                if atmname == ' C  ':
                                    prev_c_pos_tmp = pos_list_np[atm_idx]
                                    found_pos = True
                                    break
                            if not found_pos:
                                debug_global_data['positions'][prb_name]['prev_c_pos'].append(None)
                            else:
                                prev_c_pos_rot  = np.einsum('ij,j->i',prb_axis,prev_c_pos_tmp)
                                prev_c_pos = prev_c_pos_rot - prb_pos
                                debug_global_data['positions'][prb_name]['prev_c_pos'].append(prev_c_pos)

                        for atm_idx in resno_set[res]['atm']:
                            atmname = atmname_list[atm_idx]
                            atmpos_tmp  = pos_list_np[atm_idx]
                            atmpos_rot  = np.einsum('ij,j->i',prb_axis,atmpos_tmp)
                            atmpos = atmpos_rot - prb_pos
                            debug_global_data['positions'][prb_name]['atmname'][-1].append(atmname)
                            debug_global_data['positions'][prb_name]['atmpos'][-1].append(atmpos)
                            #debug_global_data['positions'][prb_name]['atmpos_orig'][-1].append(atmpos_tmp)
                            debug_global_data['positions'][prb_name]['pdbname'][-1].append(pdb_name)
                        if n_water_list_np[prb_idx] == 1:
                            #following lines are equivalent to commented block
                            watpos_tmp = neigh_water_diff_np[prb_idx][0]
                            watpos  = np.einsum('ij,j->i', prb_axis,watpos_tmp)
                            #watpos_tmp = neigh_water_diff_np[prb_idx][0] + prb_pos_tmp
                            #watpos_rot  = np.einsum('ij,j->i', prb_axis,watpos_tmp)
                            #watpos = watpos_rot - prb_pos
                            debug_global_data['positions'][prb_name]['water_pw'][-1].append(watpos)
                            #watpos_orig = neigh_water_diff_np[prb_idx][0] + prb_pos_tmp
                            #debug_global_data['positions'][prb_name]['water_orig'][-1].append(watpos_orig)
                        if n_water_ww_list_np[prb_idx] == 1:
                            watpos_tmp = neigh_water_diff_ww_np[prb_idx][0]
                            watpos  = np.einsum('ij,j->i', prb_axis,watpos_tmp)
                            debug_global_data['positions'][prb_name]['water_ww'][-1].append(watpos)
                        if n_waters_pred_np[prb_idx][0] >= score_cutoff:
                            #same logic with watpos
                            predpos_tmp = pred_np[prb_idx][0]
                            predpos  = np.einsum('ij,j->i', prb_axis,predpos_tmp)
                            debug_global_data['positions'][prb_name]['pred_pw'][-1].append(predpos)
                            #predpos_orig = pred_np[prb_idx][0] + prb_pos_tmp
                            #debug_global_data['positions'][prb_name]['pred_orig'][-1].append(predpos_orig)
                        if n_waters_pred_np[prb_idx][1] >= score_cutoff:
                            #same logic with watpos
                            predpos_tmp = pred_np[prb_idx][1]
                            predpos  = np.einsum('ij,j->i', prb_axis,predpos_tmp)
                            debug_global_data['positions'][prb_name]['pred_ww'][-1].append(predpos)
                            #predpos_orig = pred_np[prb_idx][0] + prb_pos_tmp
                            #debug_global_data['positions'][prb_name]['pred_orig'][-1].append(predpos_orig)
                                     
                

            if pdbpath_chain[1] == None:
                wat_pred_f = open('%s/%s_pred.pdb'%(result_pdb_dir,pdb_name),'w')
            else:
                wat_pred_f = open('%s/%s_%s_pred.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')                
            atmno = 0
            """
            for i, pt in enumerate(pred_all_pw):
                atmno += 1
                txt = 'HETATM%5d  O   APW U%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno%100000,i%10000,*pt, 0.0, 100*pred_n_all_pw[i])
                wat_pred_f.write(txt)
            for i, pt in enumerate(pred_all_ww):
                atmno += 1
                txt = 'HETATM%5d  O   AWW V%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno%100000,i%10000,*pt, 0.0, 100*pred_n_all_ww[i])
                wat_pred_f.write(txt)
            for i, pt in enumerate(pred_filt_pw):
                atmno += 1
                txt = 'HETATM%5d  O   FPW W%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno%100000,i%10000,*pt, 0.0, 100*pred_n_filt_pw[i])
                wat_pred_f.write(txt)
            for i, pt in enumerate(pred_filt_ww):
                atmno += 1
                txt = 'HETATM%5d  O   FWW X%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno%100000,i%10000,*pt, 0.0, 100*pred_n_filt_ww[i])
                wat_pred_f.write(txt)
            """    
            for i, pt in enumerate(pt_clust):
                atmno += 1
                txt = 'HETATM%5d  O   HOH Y%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n'%(atmno%100000,i%10000,*pt, 0.0, 100*n_clust[i])
                wat_pred_f.write(txt)
            """
            for i in range(water_pos_np.shape[0]):
                pos = water_pos_np[i]
                atmno += 1
                txt = 'HETATM%5d  O   TRU Z%4d    %8.3f%8.3f%8.3f\n'%(atmno%100000,i%10000,*pos)
                wat_pred_f.write(txt)

            for i in range(water_pos_pw_np.shape[0]):
                if n_water_list_np[i][0] > 0.5:
                    pos = water_pos_pw_np[i]
                    atmno += 1
                    txt = 'HETATM%5d  O   TPW S%4d    %8.3f%8.3f%8.3f\n'%(atmno%100000,i%10000,*pos)
                    wat_pred_f.write(txt)
            for i in range(water_pos_ww_np.shape[0]):
                if n_water_ww_list_np[i][0] > 0.5:
                    pos = water_pos_ww_np[i]
                    atmno += 1
                    txt = 'HETATM%5d  O   TWW T%4d    %8.3f%8.3f%8.3f\n'%(atmno%100000,i%10000,*pos)
                    wat_pred_f.write(txt)
            """
            wat_pred_f.close()
            pos_time_end = time.time()
            pos_time = pos_time_end - pos_time_start
            item['pos_time'] = pos_time
            #print(mem())

            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
            
            #item['pos_all'] = pred_all
            #item['pos_filt'] = pred_filt 
            #performances = {'all':[], 'filt':[], 'clust':[]}
    
    logf.close()
    timef.close()
    model.train()
