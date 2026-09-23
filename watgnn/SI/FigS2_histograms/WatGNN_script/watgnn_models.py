#charge / hyb /axis /bond is not used in ablation model

import numpy as np

from torch import nn
import torch
import dgl
import os,gc
#https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer
#SE3Transformer/se3_transformer/model

#git clone https://github.com/NVIDIA/DeepLearningExamples
#cd DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer

#docker build -t se3-transformer .
from se3_transformer import Fiber, SE3Transformer
from se3_transformer.layers import LinearSE3, NormSE3
from watgnn_visualization import pdb_dict_as_pdb, pdb_dict_as_pdb_new
from watgnn_input_preprocess import get_probe,partition_pdb_dict

#from cg2all
#torch_basics.py
EPS = 0.00001
pi = torch.tensor(np.pi)

# some basic functions
v_size = lambda v: torch.linalg.norm(v, dim=-1)
v_norm = lambda v: v / v_size(v)[..., None]

def v_nonzero(v, index=0):
    safe = torch.zeros(v.size(-1), device=v.device, dtype=v.dtype)
    safe[index] = 1.0
    #
    size = v_size(v)[..., None]
    u = torch.where(size > EPS, v, safe)
    return u

def v_norm_safe(v, index=0):
    return v_norm(v_nonzero(v, index=index))

def inner_product(v1, v2):
    return torch.sum(v1 * v2, dim=-1)

#=============================================================================
#from cg2all
#libmodel.py
class InitializationModule(nn.Module):
    #NormSE3:
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm() ──> ReLU() ──┐
    feature_in ─┤                                                      * ──> feature_out
                 └──> feature_phase ─────────────────┘
    """

    #linearSE3:
    """
    Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
    Maps a fiber to a fiber with the same degrees (channels may be different).
    No interaction between degrees, but interaction between channels.

    type-0 features (C_0 channels) ────> Linear(bias=False) ────> type-0 features (C'_0 channels)
    type-1 features (C_1 channels) ────> Linear(bias=False) ────> type-1 features (C'_1 channels)
                                                 :
    type-k features (C_k channels) ────> Linear(bias=False) ────> type-k features (C'_k channels)
    """

    def __init__(self, config):
        super().__init__()
        #
        if config['nonlinearity'] == "elu":
            nonlinearity = nn.ELU()
        elif config['nonlinearity'] == "relu":
            nonlinearity = nn.ReLU()
        elif config['nonlinearity'] == "tanh":
            nonlinearity = nn.Tanh()
        #
        linear_module = []
        if config['norm'][0]:
            linear_module.append(NormSE3(Fiber(config['fiber_init']), nonlinearity=nonlinearity))
        linear_module.append(LinearSE3(Fiber(config['fiber_init']), Fiber(config['fiber_init_pass'])))
        #
        for _ in range(config['num_linear_layers'] - 1):
            if config['norm'][0]:
                linear_module.append(NormSE3(Fiber(config['fiber_init_pass']), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(config['fiber_init_pass']), Fiber(config['fiber_init_pass'])))
        #
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, feats):
        #print(feats['0'].shape, feats['1'].shape)
        out = self.linear_module(feats)
        #print(out['0'].shape, out['1'].shape)
        return out

class InteractionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        if config['nonlinearity'] == "elu":
            nonlinearity = nn.ELU()
        elif config['nonlinearity'] == "relu":
            nonlinearity = nn.ReLU()
        elif config['nonlinearity'] == "tanh":
            nonlinearity = nn.Tanh()
        #
        self.graph_module = SE3Transformer(
            num_layers=config['num_graph_layers'],
            fiber_in=Fiber(config['fiber_init_pass']),
            fiber_hidden=Fiber(config['fiber_hidden']),
            fiber_out=Fiber(config['fiber_pass']),
            num_heads=config['num_heads'],
            channels_div=config['channels_div'],
            fiber_edge=Fiber(config['fiber_edge'] or {}),
            mid_dim=config['mid_dim'],
            norm=config['norm'][0],
            use_layer_norm=config['norm'][0],
            nonlinearity=nonlinearity,
            low_memory=config['low_memory'],
        )

    def forward(self, batch: dgl.DGLGraph, node_feats, edge_feats):
        out = self.graph_module(batch, node_feats=node_feats, edge_feats=edge_feats)
        return out

class StructureModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config['nonlinearity'] == "elu":
            nonlinearity = nn.ELU()
        elif config['nonlinearity'] == "relu":
            nonlinearity = nn.ReLU()
        elif config['nonlinearity'] == "tanh":
            nonlinearity = nn.Tanh()
        #
        linear_module = []
        #
        if config['norm'][0]:
            linear_module.append(NormSE3(Fiber(config['fiber_struct']), nonlinearity=nonlinearity))
        linear_module.append(LinearSE3(Fiber(config['fiber_struct']), Fiber(config['fiber_pass'])))
        #
        for _ in range(config['num_linear_layers'] - 2):
            if config['norm'][0]:
                linear_module.append(NormSE3(Fiber(config['fiber_pass']), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(config['fiber_pass']), Fiber(config['fiber_pass'])))
        #
        if config['norm'][0]:
            linear_module.append(NormSE3(Fiber(config['fiber_pass']), nonlinearity=nonlinearity))
        linear_module.append(LinearSE3(Fiber(config['fiber_pass']), Fiber(config['fiber_out'])))
        #
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, feats):
        out = self.linear_module(feats)
        return out
#=============================================================================

class Model(nn.Module):
    def __init__(self, _config=None):
        super().__init__()
        self.device = _config['device']
        self.embed_atm = nn.Embedding( 10, 16) # input:  10 -> output: 16 230908
        self.interval = _config['interval']
        self.loss_weight = _config['loss_weight']
        self.debug = _config['debug']
        self.n_grid = _config['n_grid']
        self.n_pt = (self.n_grid -self.n_grid//2)*(self.n_grid**2) 
        self.interaction_module    = InteractionModule(_config)
        self.structure_module      = StructureModule(_config)
        """
        res_dict = {"ALA":0 ,"ARG":1 ,"ASN":2 ,"ASP":3 ,
                    "CYS":4 ,"GLN":5 ,"GLU":6 ,"GLY":7 ,
                    "HIS":8 ,"ILE":9 ,"LEU":10,"LYS":11,
                    "MET":12,"PHE":13,"PRO":14,"SER":15,
                    "THR":16,"TRP":17,"TYR":18,"VAL":19,
                    "HID":8, "HIE":8, "HIP":8}
        #atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3}
        atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"PRB":7,"_ELSE":8}
        """
    def forward(self, pdb_dict, pdb_path=None, chain=None, save_input=False):

        device = self.device
        loss_weight = self.loss_weight 
        is_merged = False
        if 'merged' in pdb_dict.keys():
            is_merged = True

        probe_dict = get_probe(pdb_dict) #atm + probe
        pos_list_atm = pdb_dict['pos_list']
        pos_list_all = probe_dict['pos_list']
        probe_link_list = probe_dict['probe_link_list']

        n_pt = 1
        if len(pos_list_atm.shape) == 0:
            print('no atoms')
            raise ValueError
        
        #build graph with pos_list, res_list, atm_list #only node feature. no edge features yet
        #graph_dist = 5.0 (changed into all-atom graph within residue)
        graph_dist_lr = 10.0
        #data = dgl.radius_graph(pos_list, graph_dist , self_loop=False) #radius: 10.0

        edge_src_intra  = []
        edge_dst_intra  = []
        #find idxs in same residue (no probe)
        resno_list   = pdb_dict['resno_list'].detach().cpu().tolist() #probe is ignored here
        resno_set    = list(set(resno_list))
        resno_dict   = {} #{res:[idx0,idx1,...]}
        for resno in resno_set:
            resno_dict[resno] = []
        for idx, resno in enumerate(resno_list):
            resno_dict[resno].append(idx)

        for resno in resno_set:
            for idx_i in resno_dict[resno]:
                for idx_j in resno_dict[resno]:
                    if is_merged:
                        if pdb_dict['ent_list'][idx_i] != pdb_dict['ent_list'][idx_j] :
                            continue
                    if idx_i != idx_j:
                        edge_src_intra.append(idx_i)
                        edge_dst_intra.append(idx_j) 
        
        edge_src_intra_torch = torch.tensor(edge_src_intra,dtype=torch.int64)
        edge_dst_intra_torch = torch.tensor(edge_dst_intra,dtype=torch.int64)

        data = dgl.graph((edge_src_intra_torch,edge_dst_intra_torch),num_nodes = pos_list_all.shape[0]) #since probe will be included in nodes


        #add long range interaction for polar atom (no probe) 
        #====================================================================
        data_lr =  dgl.radius_graph(pos_list_atm, graph_dist_lr, self_loop=False) #check collision between predicted water positions

        #print(dist_lr.shape) # L x 1
        edge_src_lr_torch, edge_dst_lr_torch = data_lr.edges() #shape: n_edge 
        
        #save only polar edge
        #print(edge_src_lr.shape)
        edge_src_lr_np = edge_src_lr_torch.detach().cpu().numpy()
        edge_dst_lr_np = edge_dst_lr_torch.detach().cpu().numpy() 
        polar_edge_mask_np = np.zeros_like(edge_src_lr_np)

        polar_mask_list = pdb_dict['polar_mask_list'] 
        polar_mask_list_np = polar_mask_list.detach().cpu().numpy()
        for idx in range(edge_src_lr_np.shape[0]):            
            src = edge_src_lr_np[idx]
            dst = edge_dst_lr_np[idx]
            if polar_mask_list_np[src] > 0.99 and polar_mask_list_np[dst] > 0.99:
                polar_edge_mask_np[idx] = 1

        polar_edge_mask_tmp = torch.from_numpy(polar_edge_mask_np) 
        polar_edge_mask = polar_edge_mask_tmp > 0.99

        edge_src_lr_torch =edge_src_lr_torch[polar_edge_mask]
        edge_dst_lr_torch =edge_dst_lr_torch[polar_edge_mask]
        
        #print(edge_src_lr.shape)
        #add edges to data
        data.add_edges(edge_src_lr_torch,edge_dst_lr_torch)

        #add probe edges
        edge_src_probe_torch = probe_link_list[:,0]
        edge_dst_probe_torch = probe_link_list[:,1]
        data.add_edges(edge_src_probe_torch,edge_dst_probe_torch)

        data = dgl.to_simple(data)


        #"""
        #====================================================================

        probe_dict['res_list'] = probe_dict['res_list'].to(device)
        probe_dict['atm_list'] = probe_dict['atm_list'].to(device)  
        embedding_atm = self.embed_atm(probe_dict['atm_list'])   

        data.ndata["pos"] = pos_list_all
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos_list_all[edge_dst] - pos_list_all[edge_src]

        #=============================================================
        edge_feat = torch.zeros((data.num_edges(), 3), dtype=torch.float32)  #   0: intra, 1: lr there might be redundant bond in 0 and 1 2:probe

        if len(edge_src_intra_torch.shape) == 1:
            eid_intra = data.edge_ids(edge_src_intra_torch, edge_dst_intra_torch)
            edge_feat[eid_intra, 0] = 1.0
        else:
            print('no intra edges')

        if len(edge_src_lr_torch.shape) == 1:
            eid_lr = data.edge_ids(edge_src_lr_torch, edge_dst_lr_torch)
            edge_feat[eid_lr, 1] = 1.0
        else:
            print('no long-range edges')

        if len(probe_link_list.shape) == 2:
            has_edges = data.has_edges_between(probe_link_list[:, 0], probe_link_list[:, 1])
            probe_link_list = probe_link_list[has_edges]
            eid = data.edge_ids(probe_link_list[:, 0], probe_link_list[:, 1])
            edge_feat[eid, 2] = 1.0
        else:
            print('no probe_link exists')
        
        #print(edge_feat.shape)
        data.edata["edge_feat_0"] = edge_feat[..., None]
        if self.debug == True:
            data_graph_dist = dgl.shortest_dist(data)
        else:
            data_graph_dist = None

        #dsave_input
        if save_input:
            if pdb_path == None:
                pdb_name_tmp = 'input'
            else:
                pdb_name_tmp = pdb_path.split('/')[-1].split('.')[0]
                
            if chain == None:
                pdb_name = pdb_name_tmp
            else:
                pdb_name = '%s_%s'%(pdb_name_tmp,chain)
            bildpath = {}
            if not os.access('junk',0):
                os.mkdir('junk')
            bildpath['bild_axis'] = 'junk/%s_axis.bild'%pdb_name
            bildpath['bild_bond'] = 'junk/%s_bond.bild'%pdb_name
            bildpath['bild_else'] = 'junk/%s_else.bild'%pdb_name
            bildpath['bild_intra'] = 'junk/%s_intra.bild'%pdb_name
            bildpath['bild_polar'] = 'junk/%s_polar.bild'%pdb_name
            bildpath['bild_probe'] = 'junk/%s_probe.bild'%pdb_name
            bildpath_orig = {}
            bildpath_orig['bild_axis'] = 'junk/%s_orig_axis.bild'%pdb_name
            bildpath_orig['bild_bond'] = 'junk/%s_orig_bond.bild'%pdb_name
            bildpath_orig['bild_else'] = 'junk/%s_orig_else.bild'%pdb_name
            bildpath_orig['bild_intra'] = 'junk/%s_orig_intra.bild'%pdb_name
            bildpath_orig['bild_polar'] = 'junk/%s_orig_polar.bild'%pdb_name
            bildpath_orig['bild_probe'] = 'junk/%s_orig_probe.bild'%pdb_name
            pdb_dict_as_pdb_new (probe_dict, dgl_graph=data, outpath='junk/%s.pdb'%pdb_name, bildpath=bildpath)
            #pdb_dict_as_pdb_new (pdb_dict, dgl_graph=data, outpath='junk/%s_orig.pdb'%pdb_name, bildpath=bildpath_orig)
        #=============================================================
        data = data.to(device)
        edge_feats = {"0": data.edata["edge_feat_0"]}

        node_feats = {
            "0": torch.unsqueeze(torch.cat([embedding_atm], dim=1), -1), #N x 16  (changed 231011)
        } 

        out0 = self.interaction_module(data, node_feats=node_feats, edge_feats=edge_feats)
        
        feats_structure = {
            "0": out0['0'],
            "1": out0['1']
        }

        out = self.structure_module(feats_structure)
        probe_dict['n_water_list'] = probe_dict['n_water_list'].to(device)
        probe_dict['n_water_ww_list'] = probe_dict['n_water_ww_list'].to(device)
        n_water_list = torch.cat([probe_dict['n_water_list'] , probe_dict['n_water_ww_list']],dim=-1) #N_probe x 1 -> N_probe x 2
        ones = torch.ones_like(probe_dict['n_water_list'])
        n_weight_list = torch.cat([ 1.5*(ones +(4-1)*probe_dict['n_water_list']) , 0.5*(ones + (4-1)*probe_dict['n_water_ww_list'])],dim=-1)

        pred_vecs =  self.interval*out['1'] 
        pred_n    = out['0'][:,:,0] #Nx1 - compare with n_water_list
        
        loss_mask_list = probe_dict['loss_mask_list'].to(device)
        
        neigh_water_diff_pw = probe_dict['neigh_water_diff'].to(device)
        neigh_water_diff_ww = probe_dict['neigh_water_diff_ww'].to(device)
        neigh_water_diff = torch.cat([neigh_water_diff_pw , neigh_water_diff_ww],dim=-2) #N_probe x 1 x 3 -> N_probe x 2 x 3
        BCE_w =   n_weight_list

        #1. polar_mask_list * (BCEloss between pred_n ~ n_water_list) - loss_0
        loss_0_f = nn.BCEWithLogitsLoss(weight=BCE_w, reduction = 'none') #keep dimension
        loss_0_tmp = loss_0_f(pred_n,n_water_list) #no consideration of polar_mask
        #2. polar_mask_list * n_water_list * (MSE between pred_vecs, neigh_water_diff)
        #torch.sum(v,2) -> summation with 2nd dimension (0,1,2)
        d_clamp = 10.0
        d_clamp_sq = d_clamp**2
        #RMSE loss (since FAPE use L1 loss)
        loss_1_tmp = torch.clamp(torch.sqrt ( torch.sum( torch.pow( (pred_vecs - neigh_water_diff), 2), 2) + EPS**2), max=d_clamp)#Nx (n_grid -n_grid//2)*(n_grid**2)^3, new!
        #MSE loss
        loss_1_sq_tmp = torch.clamp(torch.sum( torch.pow( (pred_vecs - neigh_water_diff), 2), 2), max = d_clamp_sq)
        loss_angle_tmp = torch.abs(1.0 - inner_product(v_norm_safe(pred_vecs), v_norm_safe(neigh_water_diff)))
        loss_distance_tmp = torch.abs(v_size(pred_vecs) - v_size(neigh_water_diff))

        #anti_mask_list = 1.0 - polar_mask_list
        #print(loss_0_tmp.shape, polar_mask_list.shape)
        loss_0_N   =  EPS+2*n_pt*torch.sum(loss_mask_list) #2: now it's 2 channel
        loss_0_sum = torch.sum( torch.einsum('i,ij->ij',loss_mask_list,loss_0_tmp) )

        loss_1_N   = EPS+torch.sum( torch.einsum('i,ij->ij',loss_mask_list,n_water_list))
        loss_1_sum        = torch.sum( torch.einsum('i,ij,ij->ij',loss_mask_list,n_water_list,loss_1_tmp) )
        loss_1_sq_sum        = torch.sum( torch.einsum('i,ij,ij->ij',loss_mask_list,n_water_list,loss_1_sq_tmp) )
        loss_angle_sum    = torch.sum( torch.einsum('i,ij,ij->ij',loss_mask_list,n_water_list,loss_angle_tmp) )
        loss_distance_sum = torch.sum( torch.einsum('i,ij,ij->ij',loss_mask_list,n_water_list,loss_distance_tmp) )

        loss_0 = loss_0_sum / loss_0_N
        loss_1 = loss_1_sum / loss_1_N
        loss_1_sq = loss_1_sq_sum / loss_1_N
        loss_angle = loss_angle_sum / loss_1_N
        loss_distance = loss_distance_sum / loss_1_N

        #add following losses:
        # 1. RMSE (not MSE) V
        # 2. polar-water distance difference loss (v_cntr)
        # 3. angle difference loss (v_cntr)
        # 4. H-bond angle difference loss
        #loss_0_nonpolar = loss_0_sum_nonpolar / loss_0_N_nonpolar
        #loss_1_nonpolar = loss_1_sum_nonpolar / loss_1_N_nonpolar
        #epoch ~: l0 weighted_BCE(1:8) weight 0.05 l1 weight 1   lr 0.001     

        #loss = loss_weight['n_water']*(loss_0 ) + loss_weight['pos_RMSE']*(0.5*loss_1 +0.5*loss_1_sq+ loss_angle + loss_distance) 
        loss = loss_weight['n_water']*(loss_0 ) + loss_weight['pos_RMSE']*(1.5*loss_1 )

        return {'n_water_pred':torch.sigmoid(out['0'][:,:,0]), 'pred_vecs':pred_vecs, 'data_graph_dist':data_graph_dist}, loss, (loss_0,loss_1,loss_angle, loss_distance) #temp

