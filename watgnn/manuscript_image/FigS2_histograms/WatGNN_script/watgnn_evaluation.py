import psutil
import os, time,copy,gc
import numpy as np

import torch
import pickle
import dgl
from scipy.spatial.distance import cdist

from watgnn_input_preprocess import transform, merge_pdb_dicts, get_probe, partition_pdb_dict
#from watgnn_input import read_paths,read_dataset, read_dataset_simple
from watgnn_input_suppl import read_paths,read_paths_suppl, read_dataset, read_dataset_simple
from watgnn_visualization_fig2 import pdb_dict_as_pdb_new

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
                pass
                #continue
 
            if dataset_lig[trgidx] == None:
                is_ligand = None
            
            read_time_start = time.time()
            if is_ligand:
                pdb_dict = read_paths([pdbpath_chain,dataset_lig[trgidx]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 

            else:
                pdb_dict = read_paths([pdbpath_chain],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 
                
            pdb_dict_as_pdb_new (pdb_dict, dgl_graph=None, outpath='x.pdb',bildpath=None)  #DOES NOT APPEAR IN ACTUAL CODE
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

            for k, pdb_dict_partition in enumerate(pdb_dict_partitioned):

                n_try = 0
                max_n_try=10
                done = False
                #due to VRAM allocation problem - cannot allocate memory
                while (done == False and n_try < max_n_try):
                    try:
                        pred_list[k], loss,metric = model.forward(pdb_dict_partition, pdb_path=pdb_path, chain=pdbpath_chain[1], save_input=debug) # currently, metric = (loss_0, loss_1)
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
                
                #removing predicted sites that clash with atoms from the input molecule
                input_pos_np = pdb_dict['pos_list'].numpy()
                pred_filt_np = np.array(pred_filt)
                pred_n_filt_np = np.array(pred_n_filt)
                
                dist0 = cdist(input_pos_np, np.array(pred_filt))
                mindist = dist0.min(axis=0) #fix 260922
                no_clash_mask = ( mindist > config['clust_radius'])
                pred_n_filt = pred_n_filt_np[no_clash_mask]
                pred_filt = pred_filt_np[no_clash_mask]
                del(dist0)
                
                #clustering
                pred_filt_torch = torch.from_numpy(np.array(pred_filt))
                n_max_water = min(pred_filt_torch.shape[0], 30000) 
                print('n_max_water: ',pred_filt_torch.shape[0], 30000)
                clust_indice = [ i for i in range(n_max_water)]
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
                                     
                
            #wat_pred_f = open('wat_pred.pdb','w') #duct taping
            #this does not work since current watgnn does not use pdb_dir / pdb_name structure.
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

#from gnn_test_newset_evaluation_ablation.py

def analysis_water(config):
    def v_nonzero_np(v, index=0):
        if v_size_np < EPS:
            result = np.zeros_like(v)
            result[index] = 1.0
        else:
            result = v
        return result
    def v_size_np (v):
        return np.linalg.norm(v)
    def v_norm_np (v):
        return v / v_size_np(v)
    def v_norm_safe_np(v, index=0):
        return v_norm_np(v_nonzero_np(v, index=index))
    def get_angle(v1,v2): #0~pi
        v_norm1 = v_norm_np(v1)
        v_norm2 = v_norm_np(v2)
        cos_ang = np.inner(v_norm1,v_norm2)
        result  = np.arccos(cos_ang)
        return result
    def rad_to_deg(val):
        return val*180.0 / np.pi
    def deg_to_rad(val):
        return val* np.pi /180.0
    
    def get_grid_idx(wp_diff, axis, grid_start=-4.5, interval=4.5, n_grid=2):
        e0 = axis[0]
        e1 = axis[1]
        e2 = axis[2]
        x = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e0)-grid_start) /interval))) - n_grid//2
        y = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e1)-grid_start) /interval)))
        z = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e2)-grid_start) /interval)))
        grid_idx = x*(n_grid**2) + y*(n_grid) + z
        return grid_idx
    
    def has_hbond_wp(wp_diff, polar_vec, dist_cutoff=4.5, angle_cutoff = 90):
        angle_cutoff_rad = deg_to_rad(angle_cutoff)
        if np.linalg.norm(wp_diff) < dist_cutoff:
            ##                    ->   ->              ->
            ##get angle between   pw - pn ; polar_vec: np
            angle = get_angle(wp_diff, -1.0*polar_vec)
            if angle >= angle_cutoff_rad:
                return True
            else:
                return False
        else:
            return False
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    direc = './pdb_new'
    state_dict_dir = config['state_dict_dir']
    trainpath = 'train_new_2.txt'
    testpath = 'validation_new_2.txt'
    
    n_max_trg=300
    logpath = './gnn_water_analysis.txt'
    grid_start = config['grid_start']
    interval = config['interval']
    n_grid = config['n_grid']

    trainset = read_dataset(trainpath, direc = direc, label='train', water_cutoff = 4.5, n_max_trg = n_max_trg,
                            grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval = True)
    testset = read_dataset(testpath, direc = direc, label='test ', water_cutoff = 4.5, n_max_trg=n_max_trg,
                            grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval = True)

    log_f = open(logpath,'w')

    log_f.write('hydrogen bond examination')
    #t['pdb_dict']['water_pos'] #- water position M x 3 - torch
    #t['pdb_dict']['grid_diff'] #- grid difference: N x 4 x 3 -torch
    #t['pdb_dict']['axis_list'] # axis list: N x 3 x 3
    #h-bond criteria: protein-water: dot(axis, (water_pos - polar_pos)) > 0 (90deg)  / dist(water_pos - polar_pos) <4.5
    #                 water-water: dist < 4.5 (needs check)
    #for each water:
    #1. save grid_diff (belongs to list of [nth atom / mth grid])
    #2. check having h-bond with polar atoms / h-bond with water
    #-> can I assign every water with these 3 category? (grid cannot hold more than one water for each case 1,2,3 )
    #1. having h-bond with self (for each polar atom grid)
    #2. having h-bond with other polar atom (discriminate both 1.2 and only 2)
    #3. having h-bond with other water (check 1,3 / 2,3 / 3) <-most important part
    #for neigh_water_diff
    N_case = [0 for i in range(16)] # chmod-like, has water: 8/ h-bond with self: 4 / h-bond with other: 2 / h-bond with water: 1
    N_water_probe = {}
    #probe_case = [] # ex: [ [n_case self, n_case other, n_case water] ]

    angle_cutoff = 100.0
    dist_cutoff = 3.5
    dist_cutoff_ww = 3.5  

    for t in trainset:
        #new method: pw: dist(p,w) < 3.5A / angle(pw, -polar_vec) > 100 deg (h-bond criteria)
        #            ww: inside the grid  / have one or more w-w interaction(cutoff 3.5A) / have no p-w bond
        atm_list        = t['pdb_dict']['atm_list'].detach().cpu().numpy()
        polar_mask_list = t['pdb_dict']['polar_mask_list'].detach().cpu().numpy()
        pos_list        = t['pdb_dict']['pos_list'].detach().cpu().numpy()
        water_pos_list  = t['pdb_dict']['water_pos'].detach().cpu().numpy()
        grid_diff       = t['pdb_dict']['grid_diff'].detach().cpu().numpy()
        axis_list       = t['pdb_dict']['axis_list'].detach().cpu().numpy()
        polar_atm_idxs = []
        polar_pos_list_tmp = []
        n_probe = grid_diff.shape[1] #N x n_probe x 3
        for atm_idx in range(atm_list.shape[0]):
            if polar_mask_list[atm_idx] == 1:
                polar_atm_idxs.append(atm_idx)
                polar_pos_list_tmp.append(pos_list[atm_idx])
        polar_pos_list = np.array(polar_pos_list_tmp)
        wpdist = cdist(water_pos_list, polar_pos_list)
        wwdist = cdist(water_pos_list, water_pos_list)
        
        n_wat = wpdist.shape[0]
        n_polar = wpdist.shape[1]

        probe_watidx   = [ None for i in range(n_probe*n_polar)] #water idxs in each probe
        probe_watidx_w = [ None for i in range(n_probe*n_polar)] #water idxs in each probe
        wat_probeidx = [ [] for i in range(n_wat)] #probe idxs in each water
        wat_watidx   = [ [] for i in range(n_wat)] #water having h-bond with another water (use dist_cutoff)

        #assign water to wat_watidx
        for i_wat in range(n_wat):
            for j_wat in range(n_wat):
                if i_wat == j_wat:
                    continue
                if wwdist[i_wat][j_wat] < dist_cutoff_ww:
                    wat_watidx[i_wat].append(j_wat)


        #assign water to probe (fill probe_watidx/ wat_probeidx)
        for i_wat in range(n_wat):
            for i_polar in range(n_polar):
                atm_idx = polar_atm_idxs[i_polar]
                axis = axis_list[atm_idx]
                polar_vec = axis[0] #polar_pos - neigh_pos

                if wpdist[i_wat][i_polar] > config['water_cutoff']:
                    continue
                wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar] 
                
                #new
                hbond_self = has_hbond_wp(wp_diff, polar_vec, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)
                if not hbond_self:
                    continue

                grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                if grid_idx < 0:
                    continue
                probe_idx = n_probe*i_polar + grid_idx
                if probe_watidx[probe_idx] == None:
                    probe_watidx[probe_idx] = i_wat

                elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx[probe_idx] ][i_polar]:
                    probe_watidx[probe_idx] = i_wat

        #assign probes to water molecule            
        for i_probe in range(len(probe_watidx)):
            i_wat = probe_watidx[i_probe]
            if i_wat != None:
                wat_probeidx[i_wat].append(i_probe)

        #assign water to probe (fill probe_watidx/ wat_probeidx)
        for i_wat in range(n_wat):
            for i_polar in range(n_polar):
                atm_idx = polar_atm_idxs[i_polar]
                axis = axis_list[atm_idx]
                polar_vec = axis[0] #polar_pos - neigh_pos
                if len(wat_watidx[i_wat]) == 0:
                    continue
                #no p-w hbond allowed
                if len(wat_probeidx[i_wat]) > 0:
                    continue
                wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar]
                if np.linalg.norm(wp_diff,ord=np.inf) > config['water_cutoff']:
                    continue
                grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                if grid_idx < 0:
                    continue
                probe_idx = n_probe*i_polar + grid_idx
                if probe_watidx_w[probe_idx] == None:
                    probe_watidx_w[probe_idx] = i_wat

                elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx_w[probe_idx] ][i_polar]:
                    probe_watidx_w[probe_idx] = i_wat


        #assign probes to water molecule  (for probe_watidx_w)          
        for i_probe in range(len(probe_watidx_w)):
            i_wat = probe_watidx_w[i_probe]
            if i_wat != None:
                wat_probeidx[i_wat].append(i_probe)




        #for each probe, check following item
        #1. having h-bond with probe - hbond_self
        #2. having h-bond with other atom - hbond_other
        #3. having h-bond with water - hbond_water 
        for i_probe in range(len(probe_watidx)):
            has_water   = False
            hbond_self  = False
            hbond_other = False
            hbond_water = False
            i_wat = probe_watidx[i_probe]
            
            i_polar = i_probe // n_probe
            atm_idx = polar_atm_idxs[i_polar]
            axis = axis_list[atm_idx]
            polar_vec = axis[0] #polar_pos - neigh_pos
            
            if i_wat != None:
                wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar] 
                has_water = True
                hbond_self = has_hbond_wp(wp_diff, polar_vec, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)

                for j_probe in wat_probeidx[i_wat]:
                    if hbond_other == True:
                        break
                    j_polar = j_probe // n_probe
                    #does not count for same atom
                    if i_polar == j_polar:
                        continue 
                    atm_idx_j = polar_atm_idxs[j_polar]
                    axis_j = axis_list[atm_idx_j]
                    polar_vec_j = axis_j[0] 
                    wp_diff_j =  water_pos_list[i_wat] - polar_pos_list[j_polar]
                    hbond_other_tmp = has_hbond_wp(wp_diff_j, polar_vec_j, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)
                    if hbond_other_tmp == True:
                        hbond_other = True

                if len(wat_watidx[i_wat]) >= 1:
                    hbond_water = True

            else:
                has_water = False 

            case_idx = 8*int(has_water) + 4*int(hbond_self) + 2*int(hbond_other) + 1*int(hbond_water)
            N_case[case_idx] += 1

        #for each water, check number of probes assigned to the water
        for i_wat in range(n_wat):
            n = len(wat_probeidx[i_wat])
            if not n in N_water_probe.keys():
                N_water_probe[n] = 0
            N_water_probe[n] += 1

    for i, n in enumerate(N_case):
        print ('case ', '%2d'%i, ' : ',n)
                    

    log_f.write('#number of water in grid\n')
    n_water_cnt = [ 0.0 for i in range(12)]
    N=0.0
    for t in trainset:
        atm_list = t['pdb_dict']['atm_list']
        for i in range(atm_list.shape[0]):
            mask = float(t['pdb_dict']['polar_mask_list'][i])
            if mask > 0.01: #mask = 1
                N += 1
                n_water_analysis_int = t['pdb_dict']['n_water_analysis_int']
                for j in range(n_water_analysis_int.shape[1]):
                    if n_water_analysis_int[i][j] <12:
                        n_water = n_water_analysis_int[i][j]
                        n_water_cnt[n_water] += 1 
    
    n_probe_keys = sorted(N_water_probe.keys())
    for n in n_probe_keys:
        print('N_probe: ', "%5d"%n, " N_water : ",N_water_probe[n])

    """ #current method
        atm_list        = t['pdb_dict']['atm_list'].detach().cpu().numpy()
        polar_mask_list = t['pdb_dict']['polar_mask_list'].detach().cpu().numpy()
        pos_list        = t['pdb_dict']['pos_list'].detach().cpu().numpy()
        water_pos_list  = t['pdb_dict']['water_pos'].detach().cpu().numpy()
        grid_diff       = t['pdb_dict']['grid_diff'].detach().cpu().numpy()
        axis_list       = t['pdb_dict']['axis_list'].detach().cpu().numpy()
        polar_atm_idxs = []
        polar_pos_list_tmp = []
        n_probe = grid_diff.shape[1] #N x n_probe x 3
        for atm_idx in range(atm_list.shape[0]):
            if polar_mask_list[atm_idx] == 1:
                polar_atm_idxs.append(atm_idx)
                polar_pos_list_tmp.append(pos_list[atm_idx])
        polar_pos_list = np.array(polar_pos_list_tmp)
        wpdist = cdist(water_pos_list, polar_pos_list)
        wwdist = cdist(water_pos_list, water_pos_list)
        
        n_wat = wpdist.shape[0]
        n_polar = wpdist.shape[1]

        probe_watidx   = [ None for i in range(n_probe*n_polar)] #water idxs in each probe
        probe_watidx_w = [ None for i in range(n_probe*n_polar)] #water idxs in each probe
        wat_probeidx = [ [] for i in range(n_wat)] #probe idxs in each water
        wat_watidx   = [ [] for i in range(n_wat)] #water having h-bond with another water (use dist_cutoff)

        #assign water to probe (fill probe_watidx/ wat_probeidx)
        for i_wat in range(n_wat):
            for i_polar in range(n_polar):
                atm_idx = polar_atm_idxs[i_polar]
                axis = axis_list[atm_idx]
                polar_vec = axis[0] #polar_pos - neigh_pos

                if wpdist[i_wat][i_polar] > config['water_cutoff']:
                    continue
                wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar] 
                grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                if grid_idx < 0:
                    continue
                probe_idx = n_probe*i_polar + grid_idx
                if probe_watidx[probe_idx] == None:
                    probe_watidx[probe_idx] = i_wat

                elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx[probe_idx] ][i_polar]:
                    probe_watidx[probe_idx] = i_wat

        #assign probes to water molecule            
        for i_probe in range(len(probe_watidx)):
            i_wat = probe_watidx[i_probe]
            if i_wat != None:
                wat_probeidx[i_wat].append(i_probe)


        #assign water to wat_watidx
        for i_wat in range(n_wat):
            for j_wat in range(n_wat):
                if i_wat == j_wat:
                    continue
                if wwdist[i_wat][j_wat] < dist_cutoff_ww:
                    wat_watidx[i_wat].append(j_wat)


        #for each probe, check following item
        #1. having h-bond with probe - hbond_self
        #2. having h-bond with other atom - hbond_other
        #3. having h-bond with water - hbond_water 
        for i_probe in range(len(probe_watidx)):
            has_water   = False
            hbond_self  = False
            hbond_other = False
            hbond_water = False
            i_wat = probe_watidx[i_probe]
            
            i_polar = i_probe // n_probe
            atm_idx = polar_atm_idxs[i_polar]
            axis = axis_list[atm_idx]
            polar_vec = axis[0] #polar_pos - neigh_pos
            
            if i_wat != None:
                wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar] 
                has_water = True
                hbond_self = has_hbond_wp(wp_diff, polar_vec, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)

                for j_probe in wat_probeidx[i_wat]:
                    if hbond_other == True:
                        break
                    j_polar = j_probe // n_probe
                    #does not count for same atom
                    if i_polar == j_polar:
                        continue 
                    atm_idx_j = polar_atm_idxs[j_polar]
                    axis_j = axis_list[atm_idx_j]
                    polar_vec_j = axis_j[0] 
                    wp_diff_j =  water_pos_list[i_wat] - polar_pos_list[j_polar]
                    hbond_other_tmp = has_hbond_wp(wp_diff_j, polar_vec_j, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)
                    if hbond_other_tmp == True:
                        hbond_other = True

                if len(wat_watidx[i_wat]) >= 1:
                    hbond_water = True

            else:
                has_water = False 

            case_idx = 8*int(has_water) + 4*int(hbond_self) + 2*int(hbond_other) + 1*int(hbond_water)
            N_case[case_idx] += 1

        #for each water, check number of probes assigned to the water
        for i_wat in range(n_wat):
            n = len(wat_probeidx[i_wat])
            if not n in N_water_probe.keys():
                N_water_probe[n] = 0
            N_water_probe[n] += 1

    for i, n in enumerate(N_case):
        print ('case ', '%2d'%i, ' : ',n)
                    

    log_f.write('#number of water in grid\n')
    n_water_cnt = [ 0.0 for i in range(12)]
    N=0.0
    for t in trainset:
        atm_list = t['pdb_dict']['atm_list']
        for i in range(atm_list.shape[0]):
            mask = float(t['pdb_dict']['polar_mask_list'][i])
            if mask > 0.01: #mask = 1
                N += 1
                n_water_analysis_int = t['pdb_dict']['n_water_analysis_int']
                for j in range(n_water_analysis_int.shape[1]):
                    if n_water_analysis_int[i][j] <12:
                        n_water = n_water_analysis_int[i][j]
                        n_water_cnt[n_water] += 1 
    
    n_probe_keys = sorted(N_water_probe.keys())
    for n in n_probe_keys:
        print('N_probe: ', "%5d"%n, " N_water : ",N_water_probe[n])
    """

    n_water_avg = [ n_cnt/float(N) for n_cnt in n_water_cnt]
    for i in range(len(n_water_avg)):
        log_f.write('cnt: %3d | prob: %8.3f \n'%(i,n_water_avg[i]))

    #snippets for water cutoff decision - 4.0A (94.6% for trainset)
    #4.0: 94.6% / 4.5: 97.7% / 5.0: 98.7% / 5.5: 99.2% / 6.0: 99.5% (all atom)
    #4.0: 86.6% / 4.5: 93.4% / 5.0: 97.4% / 5.5: 98.7% / 6.0: 99.2% (O/N)

    log_f.write('#minimum distance between protein and water\n')
    mindist_total = []

    for t in trainset:
        mindist = t['pdb_dict']['mindist']
        mindist_total.extend(mindist)
    N = float(len(mindist_total))
    cutoffs = [0.5*i for i in range(21)]
    probs   = [0 for i in range(21)]
    for d in mindist_total:
        for i, cutoff in enumerate(cutoffs):
            if d < cutoffs[i]:
                probs[i] += 1.0/N
    for i in range(21):
        log_f.write('cutoff: %6.2f A | prob: %8.3f \n'%(cutoffs[i],probs[i]))

    # ~2% loss (following code won't work due to change in read_pdb, whichi now n_water_list is N x (n_grid -n_grid//2)*(n_grid**2) array and can have 0 or 1 only)
    '''
    grid_start = -4.5
    interval = 3.0
    N = 0
    for t in trainset:
        atm_list = t['pdb_dict']['atm_list']
        n_water_list = t['pdb_dict']['n_water_int_list']
        #'neigh_water_diff':torch.from_numpy(np.array(neigh_water_diff, dtype=np.float32)), # N x max_neigh x 3, saves difference between water crd and protein atom crd.
        for i in range(atm_list.shape[0]):
            if t['pdb_dict']['polar_mask_list'][i] < 0.99:
                continue
            water_grid = {}
            e0 = t['pdb_dict']['axis_list'][i][0].numpy()
            e1 = t['pdb_dict']['axis_list'][i][1].numpy()
            e2 = t['pdb_dict']['axis_list'][i][2].numpy()            
            for j in range(n_water_list[i]):
                v = t['pdb_dict']['neigh_water_diff'][i][j].numpy()
                x = max(0, min(2, int( (np.dot(v,e0)-grid_start) /interval)))
                y = max(0, min(2, int( (np.dot(v,e1)-grid_start) /interval)))
                z = max(0, min(2, int( (np.dot(v,e2)-grid_start) /interval)))
                if (x,y,z) not in water_grid.keys():
                    water_grid[ (x,y,z)] = 1
                    print('x')
                else:
                    water_grid[ (x,y,z)] += 1
                    print(t['path'], i, 'more than 2' ,'|',x,y,z,'n: ', water_grid[(x,y,z)])
    '''


    log_f.close()
   
#260818 added for S2
def analysis_water_S2(model, dataset, dataset_lig, predset, config, log_dir = 'gnn_log', log_paths = ['gnn_eval_log.txt'], result_pdb_dir = 'gnn_result',label='None' ):
    def v_nonzero_np(v, index=0):
        if v_size_np < EPS:
            result = np.zeros_like(v)
            result[index] = 1.0
        else:
            result = v
        return result
    def v_size_np (v):
        return np.linalg.norm(v)
    def v_norm_np (v):
        return v / v_size_np(v)
    def v_norm_safe_np(v, index=0):
        return v_norm_np(v_nonzero_np(v, index=index))
    def get_angle(v1,v2): #0~pi
        v_norm1 = v_norm_np(v1)
        v_norm2 = v_norm_np(v2)
        cos_ang = np.inner(v_norm1,v_norm2)
        result  = np.arccos(cos_ang)
        return result
    def rad_to_deg(val):
        return val*180.0 / np.pi
    def deg_to_rad(val):
        return val* np.pi /180.0
    
    def get_grid_idx(wp_diff, axis, grid_start=-4.5, interval=4.5, n_grid=2):
        e0 = axis[0]
        e1 = axis[1]
        e2 = axis[2]
        x = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e0)-grid_start) /interval))) - n_grid//2
        y = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e1)-grid_start) /interval)))
        z = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e2)-grid_start) /interval)))
        grid_idx = x*(n_grid**2) + y*(n_grid) + z
        return grid_idx
    
    def has_hbond_wp(wp_diff, polar_vec, dist_cutoff=4.5, angle_cutoff = 90):
        angle_cutoff_rad = deg_to_rad(angle_cutoff)
        if np.linalg.norm(wp_diff) < dist_cutoff:
            ##                    ->   ->              ->
            ##get angle between   pw - pn ; polar_vec: np
            angle = get_angle(wp_diff, -1.0*polar_vec)
            if angle >= angle_cutoff_rad:
                return True
            else:
                return False
        else:
            return False
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if not os.access(result_pdb_dir,0):
        os.mkdir(result_pdb_dir)

    logpath  = log_paths[0]
    logpath2 = log_paths[1]
    grid_start = config['grid_start']  #start point of grid (-4.5A from atom crd)
    interval   = config['interval']     #grid interval
    n_grid     = config['n_grid']         #maximum number of grid
    water_cutoff = config['water_cutoff']
    debug = config['debug']    
    score_cutoff = config['score_cutoff']   
    performances = {'all':{}, 'filt':{}, 'clust':{}}

    log_f  = open(logpath,'w')
    log_f2 = open(logpath2,'w')
    log_f.write('hydrogen bond examination')
    #t['pdb_dict']['water_pos'] #- water position M x 3 - torch
    #t['pdb_dict']['grid_diff'] #- grid difference: N x 4 x 3 -torch
    #t['pdb_dict']['axis_list'] # axis list: N x 3 x 3
    #h-bond criteria: protein-water: dot(axis, (water_pos - polar_pos)) > 0 (90deg)  / dist(water_pos - polar_pos) <4.5
    #                 water-water: dist < 4.5 (needs check)
    #for each water:
    #1. save grid_diff (belongs to list of [nth atom / mth grid])
    #2. check having h-bond with polar atoms / h-bond with water
    #-> can I assign every water with these 3 category? (grid cannot hold more than one water for each case 1,2,3 )
    """
    #old
    #1. having h-bond with self (for each polar atom grid)
    #2. having h-bond with other polar atom (discriminate both 1.2 and only 2)
    #3. having h-bond with other water (check 1,3 / 2,3 / 3) <-most important part
    """
    # A. # of current probe-eligible water
    # B. #number of water might be eligible if carbon-probe is adopted
    # C. #total number of crystallographic water


    #for neigh_water_diff
    N_case = [0 for i in range(16)] # chmod-like, has water: 8/ h-bond with self: 4 / h-bond with other: 2 / h-bond with water: 1
    N_water_probe = {}
    #probe_case = [] # ex: [ [n_case self, n_case other, n_case water] ]

    angle_cutoff = 100.0
    dist_cutoff = 3.5
    dist_cutoff_ww = 3.5  
    cutoffs = [0.1*i for i in range(101)]

    N_carbon_histogram = [ 0 for i in range(101)] #minimum distance between carbon atom and matching water
    N_polar_histogram  = [ 0 for i in range(101)] #minimum distance between polar atom and matching water
    N_polar_elig_pw_histogram   = [ 0 for i in range(101)] #minimum distance between polar atom and matching water (p-w)
    N_polar_elig_ww_histogram  = [ 0 for i in range(101)] #minimum distance between polar atom and matching water (w-w)
    N_total = 0

    """
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    """

    state_dict_dir = config['state_dict_dir']
    
    is_ligand = False
    if dataset_lig == None:
        is_ligand = False
    else:
        is_ligand = True
    debug_global_data = {}
    with torch.no_grad():
        for trgidx, pdbpath_chain in enumerate(dataset):
            
            if dataset_lig[trgidx] == None:
                is_ligand = None
            
            read_time_start = time.time()
            if is_ligand:
                pdb_dict = read_paths([pdbpath_chain,dataset_lig[trgidx]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 

            else:
                pdb_dict = read_paths([pdbpath_chain],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 


            pdb_name = pdbpath_chain[0].split('/')[-1].split('.')[0]
            #chainname = predset[trgidx].split('/')[-1].split('.')[0].split('_')[-1]
            wat_pw_f = None
            wat_ww_f = None
            wat_cw_f = None
            wat_none_f = None

            if pdbpath_chain[1] == None:
                wat_pw_f = open('%s/%s_pw.pdb'%(result_pdb_dir,pdb_name),'w')
                wat_ww_f = open('%s/%s_ww.pdb'%(result_pdb_dir,pdb_name),'w')
                wat_cw_f = open('%s/%s_cw.pdb'%(result_pdb_dir,pdb_name),'w')
                wat_none_f = open('%s/%s_none.pdb'%(result_pdb_dir,pdb_name),'w')
            else:
                wat_pw_f = open('%s/%s_%s_pw.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')
                wat_ww_f = open('%s/%s_%s_ww.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')
                wat_cw_f = open('%s/%s_%s_cw.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')
                wat_none_f = open('%s/%s_%s_none.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')


            #new method: pw: dist(p,w) < 3.5A / angle(pw, -polar_vec) > 100 deg (h-bond criteria)
            #            ww: inside the grid  / have one or more w-w interaction(cutoff 3.5A) / have no p-w bond
            atm_list        = pdb_dict['atm_list'].detach().cpu().numpy() #atom type embedding
            polar_mask_list = pdb_dict['polar_mask_list'].detach().cpu().numpy()
            pos_list        = pdb_dict['pos_list'].detach().cpu().numpy()
            water_pos_list  = pdb_dict['water_pos'].detach().cpu().numpy()
            grid_diff       = pdb_dict['grid_diff'].detach().cpu().numpy()
            axis_list       = pdb_dict['axis_list'].detach().cpu().numpy()
            polar_atm_idxs = []
            polar_pos_list_tmp = []

            carbon_atm_idxs = []
            carbon_pos_list_tmp = []

            n_probe = grid_diff.shape[1] #N x n_probe x 3
            for atm_idx in range(atm_list.shape[0]):
                if polar_mask_list[atm_idx] == 1:
                    polar_atm_idxs.append(atm_idx)
                    polar_pos_list_tmp.append(pos_list[atm_idx])
                if atm_list[atm_idx] == 0: #carbon
                    carbon_atm_idxs.append(atm_idx)
                    carbon_pos_list_tmp.append(pos_list[atm_idx])
            polar_pos_list = np.array(polar_pos_list_tmp)
            carbon_pos_list = np.array(carbon_pos_list_tmp)
            print(trgidx)
            if water_pos_list.shape[0] == 0:
                wcdist = np.array([[999.99]])
                wpdist = np.array([[999.99]])
                wwdist = np.array([[999.99]])
                print("no crystallographic water")
            else:
                if carbon_pos_list.shape[0] == 0:
                    wcdist = np.array([[999.9] for i in range(water_pos_list.shape[0])])
                    print("no carbon")
                else:
                    wcdist = cdist(water_pos_list, carbon_pos_list)      

                if polar_pos_list.shape[0] == 0:
                    wpdist = np.array([[999.9] for i in range(water_pos_list.shape[0])])
                    print("no probe eligible atoms")
                else:
                    wpdist = cdist(water_pos_list, polar_pos_list)

                wwdist = cdist(water_pos_list, water_pos_list)
        
            n_wat = water_pos_list.shape[0]
            n_polar = polar_pos_list.shape[0]
            n_carbon = carbon_pos_list.shape[0]
            print(n_wat,n_polar,n_carbon)
            log_f2.write('pdbpath_chain %s %s n_wat %10d n_polar %10d n_carbon %12d\n'%(pdbpath_chain[0],pdbpath_chain[1],n_wat,n_polar,n_carbon))

            probe_watidx   = [ None for i in range(n_probe*n_polar)] #water idxs in each probe #probe -> p-w water id
            probe_watidx_w = [ None for i in range(n_probe*n_polar)] #water idxs in each probe #probe -> w-w water id

            carbon_mindist = [ 999.99 for i in range(n_wat)] #minimum distance between carbon atom and matching water
            polar_mindist  = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water
            polar_mindist_elig_pw   = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water (p-w)
            polar_mindist_elig_ww   = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water (w-w)


            wat_probeidx = [ [] for i in range(n_wat)] #probe idxs in each water
            wat_watidx   = [ [] for i in range(n_wat)] #water having h-bond with another water (use dist_cutoff)

            #assign water to wat_watidx
            for i_wat in range(n_wat):
                for j_wat in range(n_wat):
                    if i_wat == j_wat:
                        continue
                    if wwdist[i_wat][j_wat] < dist_cutoff_ww:
                        wat_watidx[i_wat].append(j_wat)

            #water-carbon distance
            for i_wat in range(n_wat):
                carbon_mindist[i_wat] =  np.amin(wcdist[i_wat,:]) 

            #water-polar distance
            for i_wat in range(n_wat):
                polar_mindist[i_wat] =  np.amin(wpdist[i_wat,:]) 

            #assign water to probe (fill probe_watidx/ wat_probeidx)
            for i_wat in range(n_wat):
                for i_polar in range(n_polar):
                    atm_idx = polar_atm_idxs[i_polar]
                    axis = axis_list[atm_idx]
                    polar_vec = axis[0] #polar_pos - neigh_pos

                    if wpdist[i_wat][i_polar] > config['water_cutoff']:
                        continue
                    wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar] 
                
                    #new
                    hbond_self = has_hbond_wp(wp_diff, polar_vec, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)
                    if not hbond_self:
                        continue

                    grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                    if grid_idx < 0:
                        continue
                    probe_idx = n_probe*i_polar + grid_idx
                    if probe_watidx[probe_idx] == None:
                        probe_watidx[probe_idx] = i_wat

                    elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx[probe_idx] ][i_polar]:
                        probe_watidx[probe_idx] = i_wat
                        polar_mindist_elig_pw[i_wat] = wpdist[i_wat][i_polar]

            #assign probes to water molecule            
            for i_probe in range(len(probe_watidx)):
                i_polar = i_probe//n_probe
                i_wat = probe_watidx[i_probe]
                if i_wat != None:
                    wat_probeidx[i_wat].append(i_probe)
                    polar_mindist_elig_pw[i_wat] = wpdist[i_wat][i_polar]

            #assign water to probe (fill probe_watidx/ wat_probeidx)
            for i_wat in range(n_wat):
                for i_polar in range(n_polar):
                    atm_idx = polar_atm_idxs[i_polar]
                    axis = axis_list[atm_idx]
                    polar_vec = axis[0] #polar_pos - neigh_pos
                    if len(wat_watidx[i_wat]) == 0:
                        continue
                    #no p-w hbond allowed
                    if len(wat_probeidx[i_wat]) > 0:
                        continue
                    wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar]
                    if np.linalg.norm(wp_diff,ord=np.inf) > config['water_cutoff']:
                        continue
                    grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                    if grid_idx < 0:
                        continue
                    probe_idx = n_probe*i_polar + grid_idx
                    if probe_watidx_w[probe_idx] == None:
                        probe_watidx_w[probe_idx] = i_wat

                    elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx_w[probe_idx] ][i_polar]:
                        probe_watidx_w[probe_idx] = i_wat


            #assign probes to water molecule  (for probe_watidx_w)          
            for i_probe in range(len(probe_watidx_w)):
                i_polar = i_probe//n_probe
                i_wat = probe_watidx_w[i_probe]
                if i_wat != None:
                    wat_probeidx[i_wat].append(i_probe)
                    polar_mindist_elig_ww[i_wat] = wpdist[i_wat][i_polar]


            N_total += n_wat
            for i_wat in range(n_wat):
                #if water_pos_list.shape[0] == 0 :
                
                txt = 'HETATM%5d  O   HOH X%4d    %8.3f%8.3f%8.3f\n'%(i_wat%100000,i_wat%10000,*water_pos_list[i_wat])
                N_polar_idx = max(0, min(100, int(polar_mindist[i_wat]/0.1) ))
                N_polar_elig_pw_idx = max(0, min(100, int(polar_mindist_elig_pw[i_wat]/0.1) ))
                N_polar_elig_ww_idx = max(0, min(100, int(polar_mindist_elig_ww[i_wat]/0.1) ))
                N_carbon_idx = max(0, min(100, int(carbon_mindist[i_wat]/0.1) ))

                N_polar_histogram[N_polar_idx] += 1 #minimum distance between polar atom and matching water

                if polar_mindist_elig_pw[i_wat] > config['water_cutoff']: #has no eligible pw probe


                    if polar_mindist_elig_ww[i_wat] > config['water_cutoff']: #has no eligible ww probe, checked after pw probe
                        N_carbon_histogram[N_carbon_idx] += 1
                        if carbon_mindist[i_wat] > config['water_cutoff']: #has no neighboring "even" non-probe eligible atoms.
                            wat_none_f.write(txt)
                        else: #neighboring "only" non-probe eligible atoms.
                            wat_cw_f.write(txt)

                    else: #has eligible ww probe, while having no pw probe
                        N_polar_elig_ww_histogram[N_polar_elig_ww_idx] += 1
                        wat_ww_f.write(txt)

                else: #has eligible pw probe
                    N_polar_elig_pw_histogram[N_polar_elig_pw_idx] += 1 #minimum distance between polar atom and matching water remember! cut at water_cutoff!
                    wat_pw_f.write(txt)

            wat_pw_f.close()
            wat_ww_f.close()
            wat_cw_f.close()
            wat_none_f.close()
            del(wat_pw_f)
            del(wat_ww_f)
            del(wat_cw_f)
            del(wat_none_f)                
            #atmno = 0
            #for i, pt in enumerate(pred_all_pw):
            #    atmno += 1
            #    wat_pred_f.write(txt)
   
        log_f.write('# of total water molecules in the set: %12d\n'%N_total)

        log_f.write('#histogram mindist_polar : minimum distance between polar atom and water molecule (ignore h-bond eligibility\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_polar_histogram[i], N_polar_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_polar_histogram[100], N_polar_histogram[100]/N_total ))

        log_f.write('#histogram mindist_pw : minimum distance between eligible polar atom and water molecule (p-w)\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_polar_elig_pw_histogram[i], N_polar_elig_pw_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_polar_elig_pw_histogram[100], N_polar_elig_pw_histogram[100]/N_total ))

        log_f.write('#histogram mindist_ww : minimum distance between eligible polar atom and water molecule (w-w, no p-w)\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_polar_elig_ww_histogram[i], N_polar_elig_ww_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_polar_elig_ww_histogram[100], N_polar_elig_ww_histogram[100]/N_total ))

        log_f.write('#histogram mindist_cw : minimum distance between carbon atom and water molecule (no p-w / w-w)\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_carbon_histogram[i], N_carbon_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_carbon_histogram[100], N_carbon_histogram[100]/N_total ))

        log_f.close()
        
        
        
def analysis_water_S8(model, dataset, dataset_lig, predset, config, log_dir = 'gnn_log', log_paths = ['gnn_eval_log.txt'], result_pdb_dir = 'gnn_result',label='None',water_scorecut=0.1 ):
    def v_nonzero_np(v, index=0):
        if v_size_np < EPS:
            result = np.zeros_like(v)
            result[index] = 1.0
        else:
            result = v
        return result
    def v_size_np (v):
        return np.linalg.norm(v)
    def v_norm_np (v):
        return v / v_size_np(v)
    def v_norm_safe_np(v, index=0):
        return v_norm_np(v_nonzero_np(v, index=index))
    def get_angle(v1,v2): #0~pi
        v_norm1 = v_norm_np(v1)
        v_norm2 = v_norm_np(v2)
        cos_ang = np.inner(v_norm1,v_norm2)
        result  = np.arccos(cos_ang)
        return result
    def rad_to_deg(val):
        return val*180.0 / np.pi
    def deg_to_rad(val):
        return val* np.pi /180.0
    
    def get_grid_idx(wp_diff, axis, grid_start=-4.5, interval=4.5, n_grid=2):
        e0 = axis[0]
        e1 = axis[1]
        e2 = axis[2]
        x = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e0)-grid_start) /interval))) - n_grid//2
        y = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e1)-grid_start) /interval)))
        z = max(0, min( (n_grid-1), int( (np.dot(wp_diff,e2)-grid_start) /interval)))
        grid_idx = x*(n_grid**2) + y*(n_grid) + z
        return grid_idx
    
    def has_hbond_wp(wp_diff, polar_vec, dist_cutoff=4.5, angle_cutoff = 90):
        angle_cutoff_rad = deg_to_rad(angle_cutoff)
        if np.linalg.norm(wp_diff) < dist_cutoff:
            ##                    ->   ->              ->
            ##get angle between   pw - pn ; polar_vec: np
            angle = get_angle(wp_diff, -1.0*polar_vec)
            if angle >= angle_cutoff_rad:
                return True
            else:
                return False
        else:
            return False
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if not os.access(result_pdb_dir,0):
        os.mkdir(result_pdb_dir)

    logpath  = log_paths[0]
    logpath2 = log_paths[1]
    grid_start = config['grid_start']  #start point of grid (-4.5A from atom crd)
    interval   = config['interval']     #grid interval
    n_grid     = config['n_grid']         #maximum number of grid
    water_cutoff = config['water_cutoff']
    debug = config['debug']    
    score_cutoff = config['score_cutoff']   
    performances = {'all':{}, 'filt':{}, 'clust':{}}

    log_f  = open(logpath,'w')
    log_f2 = open(logpath2,'w')
    log_f.write('hydrogen bond examination')
    #t['pdb_dict']['water_pos'] #- water position M x 3 - torch
    #t['pdb_dict']['grid_diff'] #- grid difference: N x 4 x 3 -torch
    #t['pdb_dict']['axis_list'] # axis list: N x 3 x 3
    #h-bond criteria: protein-water: dot(axis, (water_pos - polar_pos)) > 0 (90deg)  / dist(water_pos - polar_pos) <4.5
    #                 water-water: dist < 4.5 (needs check)
    #for each water:
    #1. save grid_diff (belongs to list of [nth atom / mth grid])
    #2. check having h-bond with polar atoms / h-bond with water
    #-> can I assign every water with these 3 category? (grid cannot hold more than one water for each case 1,2,3 )
    """
    #old
    #1. having h-bond with self (for each polar atom grid)
    #2. having h-bond with other polar atom (discriminate both 1.2 and only 2)
    #3. having h-bond with other water (check 1,3 / 2,3 / 3) <-most important part
    """
    # A. # of current probe-eligible water
    # B. #number of water might be eligible if carbon-probe is adopted
    # C. #total number of crystallographic water


    #for neigh_water_diff
    N_case = [0 for i in range(16)] # chmod-like, has water: 8/ h-bond with self: 4 / h-bond with other: 2 / h-bond with water: 1
    N_water_probe = {}
    #probe_case = [] # ex: [ [n_case self, n_case other, n_case water] ]

    angle_cutoff = 100.0
    dist_cutoff = 3.5
    dist_cutoff_ww = 3.5  
    cutoffs = [0.1*i for i in range(101)]

    N_carbon_histogram = [ 0 for i in range(101)] #minimum distance between carbon atom and matching water
    N_polar_histogram  = [ 0 for i in range(101)] #minimum distance between polar atom and matching water
    N_polar_elig_pw_histogram   = [ 0 for i in range(101)] #minimum distance between polar atom and matching water (p-w)
    N_polar_elig_ww_histogram  = [ 0 for i in range(101)] #minimum distance between polar atom and matching water (w-w)
    N_total = 0

    """
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    """

    state_dict_dir = config['state_dict_dir']
    
    is_ligand = False
    if dataset_lig == None:
        is_ligand = False
    else:
        is_ligand = True
    debug_global_data = {}
    with torch.no_grad():
        for trgidx, pdbpath_chain in enumerate(dataset):
            predpath = predset[trgidx]
            if dataset_lig[trgidx] == None:
                is_ligand = None
            
            read_time_start = time.time()
            if is_ligand:
                pdb_dict = read_paths_revision4([pdbpath_chain,dataset_lig[trgidx]],predpath,water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, water_scorecut=water_scorecut ,is_eval=True)
                #pdb_dict = read_paths([pdbpath_chain,dataset_lig[trgidx]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 

            else:
                pdb_dict = read_paths_revision4([pdbpath_chain],predpath,water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, water_scorecut=water_scorecut ,is_eval=True)
                #pdb_dict = read_paths([pdbpath_chain],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=True) 


            pdb_name = pdbpath_chain[0].split('/')[-1].split('.')[0]
            #chainname = predset[trgidx].split('/')[-1].split('.')[0].split('_')[-1]
            wat_pw_f = None
            wat_ww_f = None
            wat_cw_f = None
            wat_none_f = None

            if pdbpath_chain[1] == None:
                wat_pw_f = open('%s/%s_pw.pdb'%(result_pdb_dir,pdb_name),'w')
                wat_ww_f = open('%s/%s_ww.pdb'%(result_pdb_dir,pdb_name),'w')
                wat_cw_f = open('%s/%s_cw.pdb'%(result_pdb_dir,pdb_name),'w')
                wat_none_f = open('%s/%s_none.pdb'%(result_pdb_dir,pdb_name),'w')
            else:
                wat_pw_f = open('%s/%s_%s_pw.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')
                wat_ww_f = open('%s/%s_%s_ww.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')
                wat_cw_f = open('%s/%s_%s_cw.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')
                wat_none_f = open('%s/%s_%s_none.pdb'%(result_pdb_dir,pdb_name,pdbpath_chain[1]),'w')


            #new method: pw: dist(p,w) < 3.5A / angle(pw, -polar_vec) > 100 deg (h-bond criteria)
            #            ww: inside the grid  / have one or more w-w interaction(cutoff 3.5A) / have no p-w bond
            atm_list        = pdb_dict['atm_list'].detach().cpu().numpy() #atom type embedding
            polar_mask_list = pdb_dict['polar_mask_list'].detach().cpu().numpy()
            pos_list        = pdb_dict['pos_list'].detach().cpu().numpy()
            water_pos_list  = pdb_dict['water_pos'].detach().cpu().numpy()
            grid_diff       = pdb_dict['grid_diff'].detach().cpu().numpy()
            axis_list       = pdb_dict['axis_list'].detach().cpu().numpy()
            polar_atm_idxs = []
            polar_pos_list_tmp = []

            carbon_atm_idxs = []
            carbon_pos_list_tmp = []

            n_probe = grid_diff.shape[1] #N x n_probe x 3
            for atm_idx in range(atm_list.shape[0]):
                if polar_mask_list[atm_idx] == 1:
                    polar_atm_idxs.append(atm_idx)
                    polar_pos_list_tmp.append(pos_list[atm_idx])
                if atm_list[atm_idx] == 0: #carbon
                    carbon_atm_idxs.append(atm_idx)
                    carbon_pos_list_tmp.append(pos_list[atm_idx])
            polar_pos_list = np.array(polar_pos_list_tmp)
            carbon_pos_list = np.array(carbon_pos_list_tmp)
            print(trgidx)
            if water_pos_list.shape[0] == 0:
                wcdist = np.array([[999.99]])
                wpdist = np.array([[999.99]])
                wwdist = np.array([[999.99]])
                widist = np.array([[999.99]])
                print("no crystallographic water")
            else:
                if carbon_pos_list.shape[0] == 0:
                    wcdist = np.array([[999.9] for i in range(water_pos_list.shape[0])])
                    print("no carbon")
                else:
                    wcdist = cdist(water_pos_list, carbon_pos_list)      

                if polar_pos_list.shape[0] == 0:
                    wpdist = np.array([[999.9] for i in range(water_pos_list.shape[0])])
                    print("no probe eligible atoms")
                else:
                    wpdist = cdist(water_pos_list, polar_pos_list)
                    
                if pos_list.shape[0] == 0:
                    widist = np.array([[999.9] for i in range(water_pos_list.shape[0])])
                    print("no input atom")
                else:
                    widist = cdist(water_pos_list, pos_list)    

                wwdist = cdist(water_pos_list, water_pos_list)
        
            n_wat = water_pos_list.shape[0]
            n_polar = polar_pos_list.shape[0]
            n_carbon = carbon_pos_list.shape[0]
            print(n_wat,n_polar,n_carbon)
            log_f2.write('pdbpath_chain %s %s n_wat %10d n_polar %10d n_carbon %12d\n'%(pdbpath_chain[0],pdbpath_chain[1],n_wat,n_polar,n_carbon))

            probe_watidx   = [ None for i in range(n_probe*n_polar)] #water idxs in each probe #probe -> p-w water id
            probe_watidx_w = [ None for i in range(n_probe*n_polar)] #water idxs in each probe #probe -> w-w water id

            carbon_mindist = [ 999.99 for i in range(n_wat)] #minimum distance between carbon atom and matching water
            polar_mindist  = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water
            polar_mindist_elig_pw   = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water (p-w)
            polar_mindist_elig_ww   = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water (w-w)
            input_mindist  = [ 999.99 for i in range(n_wat)] #minimum distance between input atom and matching water
            ww_mindist  = [ 999.99 for i in range(n_wat)] #minimum distance between polar atom and matching water

            wat_probeidx = [ [] for i in range(n_wat)] #probe idxs in each water
            wat_watidx   = [ [] for i in range(n_wat)] #water having h-bond with another water (use dist_cutoff)

            #assign water to wat_watidx
            for i_wat in range(n_wat):
                for j_wat in range(n_wat):
                    if i_wat == j_wat:
                        continue
                    if wwdist[i_wat][j_wat] < dist_cutoff_ww:
                        wat_watidx[i_wat].append(j_wat)

            #water-carbon distance
            for i_wat in range(n_wat):
                carbon_mindist[i_wat] =  np.amin(wcdist[i_wat,:]) 

            #water-polar distance
            for i_wat in range(n_wat):
                polar_mindist[i_wat] =  np.amin(wpdist[i_wat,:]) 
                
            #water-input distance
            for i_wat in range(n_wat):
                input_mindist[i_wat] =  np.amin(widist[i_wat,:]) 
                
            #water-water distance
            for i_wat in range(n_wat):
                wwdist[i_wat,i_wat] = 9999.99 # ignore self matching
                ww_mindist[i_wat] =  np.amin(wwdist[i_wat,:]) 

            #assign water to probe (fill probe_watidx/ wat_probeidx)
            for i_wat in range(n_wat):
                for i_polar in range(n_polar):
                    atm_idx = polar_atm_idxs[i_polar]
                    axis = axis_list[atm_idx]
                    polar_vec = axis[0] #polar_pos - neigh_pos

                    if wpdist[i_wat][i_polar] > config['water_cutoff']:
                        continue
                    wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar] 
                
                    #new
                    hbond_self = has_hbond_wp(wp_diff, polar_vec, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)
                    if not hbond_self:
                        continue

                    grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                    if grid_idx < 0:
                        continue
                    probe_idx = n_probe*i_polar + grid_idx
                    if probe_watidx[probe_idx] == None:
                        probe_watidx[probe_idx] = i_wat

                    elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx[probe_idx] ][i_polar]:
                        probe_watidx[probe_idx] = i_wat
                        polar_mindist_elig_pw[i_wat] = wpdist[i_wat][i_polar]

            #assign probes to water molecule            
            for i_probe in range(len(probe_watidx)):
                i_polar = i_probe//n_probe
                i_wat = probe_watidx[i_probe]
                if i_wat != None:
                    wat_probeidx[i_wat].append(i_probe)
                    polar_mindist_elig_pw[i_wat] = wpdist[i_wat][i_polar]

            #assign water to probe (fill probe_watidx/ wat_probeidx)
            for i_wat in range(n_wat):
                for i_polar in range(n_polar):
                    atm_idx = polar_atm_idxs[i_polar]
                    axis = axis_list[atm_idx]
                    polar_vec = axis[0] #polar_pos - neigh_pos
                    if len(wat_watidx[i_wat]) == 0:
                        continue
                    #no p-w hbond allowed
                    if len(wat_probeidx[i_wat]) > 0:
                        continue
                    wp_diff = water_pos_list[i_wat] - polar_pos_list[i_polar]
                    if np.linalg.norm(wp_diff,ord=np.inf) > config['water_cutoff']:
                        continue
                    grid_idx = get_grid_idx(wp_diff, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
                    if grid_idx < 0:
                        continue
                    probe_idx = n_probe*i_polar + grid_idx
                    if probe_watidx_w[probe_idx] == None:
                        probe_watidx_w[probe_idx] = i_wat

                    elif wpdist[i_wat][i_polar] < wpdist[ probe_watidx_w[probe_idx] ][i_polar]:
                        probe_watidx_w[probe_idx] = i_wat


            #assign probes to water molecule  (for probe_watidx_w)          
            for i_probe in range(len(probe_watidx_w)):
                i_polar = i_probe//n_probe
                i_wat = probe_watidx_w[i_probe]
                if i_wat != None:
                    wat_probeidx[i_wat].append(i_probe)
                    polar_mindist_elig_ww[i_wat] = wpdist[i_wat][i_polar]


            N_total += n_wat
            for i_wat in range(n_wat):
                #if water_pos_list.shape[0] == 0 :
                
                txt = 'HETATM%5d  O   HOH X%4d    %8.3f%8.3f%8.3f\n'%(i_wat%100000,i_wat%10000,*water_pos_list[i_wat])
                N_polar_idx = max(0, min(100, int(polar_mindist[i_wat]/0.1) ))
                N_polar_elig_pw_idx = max(0, min(100, int(polar_mindist_elig_pw[i_wat]/0.1) ))
                N_polar_elig_ww_idx = max(0, min(100, int(polar_mindist_elig_ww[i_wat]/0.1) ))
                N_carbon_idx = max(0, min(100, int(carbon_mindist[i_wat]/0.1) ))
                N_input_idx = max(0, min(100, int(input_mindist[i_wat]/0.1) ))
                N_water_idx = max(0, min(100, int(ww_mindist[i_wat]/0.1) ))
                
                N_polar_histogram[N_polar_idx] += 1 #minimum distance between polar atom and matching water
                N_input_histogram[N_input_idx] += 1 #minimum distance between input atom and matching water
                N_water_histogram[N_water_idx] += 1 #minimum distance between water and matching water
                
                if polar_mindist_elig_pw[i_wat] > config['water_cutoff']: #has no eligible pw probe


                    if polar_mindist_elig_ww[i_wat] > config['water_cutoff']: #has no eligible ww probe, checked after pw probe
                        N_carbon_histogram[N_carbon_idx] += 1
                        if carbon_mindist[i_wat] > config['water_cutoff']: #has no neighboring "even" non-probe eligible atoms.
                            wat_none_f.write(txt)
                        else: #neighboring "only" non-probe eligible atoms.
                            wat_cw_f.write(txt)

                    else: #has eligible ww probe, while having no pw probe
                        N_polar_elig_ww_histogram[N_polar_elig_ww_idx] += 1
                        wat_ww_f.write(txt)

                else: #has eligible pw probe
                    N_polar_elig_pw_histogram[N_polar_elig_pw_idx] += 1 #minimum distance between polar atom and matching water remember! cut at water_cutoff!
                    wat_pw_f.write(txt)

            wat_pw_f.close()
            wat_ww_f.close()
            wat_cw_f.close()
            wat_none_f.close()
            del(wat_pw_f)
            del(wat_ww_f)
            del(wat_cw_f)
            del(wat_none_f)                
            #atmno = 0
            #for i, pt in enumerate(pred_all_pw):
            #    atmno += 1
            #    wat_pred_f.write(txt)
   
        log_f.write('# of total water molecules in the set: %12d\n'%N_total)

        log_f.write('#histogram mindist_polar : minimum distance between polar atom and water molecule (ignore h-bond eligibility\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_polar_histogram[i], N_polar_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_polar_histogram[100], N_polar_histogram[100]/N_total ))

        log_f.write('#histogram mindist_pw : minimum distance between eligible polar atom and water molecule (p-w)\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_polar_elig_pw_histogram[i], N_polar_elig_pw_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_polar_elig_pw_histogram[100], N_polar_elig_pw_histogram[100]/N_total ))

        log_f.write('#histogram mindist_ww : minimum distance between eligible polar atom and water molecule (w-w, no p-w)\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_polar_elig_ww_histogram[i], N_polar_elig_ww_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_polar_elig_ww_histogram[100], N_polar_elig_ww_histogram[100]/N_total ))

        log_f.write('#histogram mindist_cw : minimum distance between carbon atom and water molecule (no p-w / w-w)\n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_carbon_histogram[i], N_carbon_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_carbon_histogram[100], N_carbon_histogram[100]/N_total ))

        log_f.write('#histogram mindist_iw : minimum distance between input atom and water molecule \n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_input_histogram[i], N_input_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_input_histogram[100], N_input_histogram[100]/N_total ))

        log_f.write('#histogram mindist_water : minimum distance between water molecule and another water molecule \n')
        for i in range(100):
            log_f.write ("%5.2fA - %5.2fA : %12d %8.3f\n"%(i*0.1 , (i+1)*0.1, N_water_histogram[i], N_water_histogram[i]/N_total ))
        log_f.write ("%5.2fA - inf    : %12d %8.3f\n"%(100*0.1 , N_water_histogram[100], N_water_histogram[100]/N_total ))
        
        log_f.close()