import os, glob, time,copy,random
from watgnn_features import bond_dict, polar_vec_dict, aux_vec_dict, charge_dict, hyb_dict 
from scipy.spatial.distance import cdist
import numpy as np

from torch import nn,optim
import torch
import pickle
import dgl

def pdb_dict_as_pdb_new (pdb_dict, dgl_graph=None, outpath='x.pdb',bildpath=None):
    pdb_f = open(outpath,'w')
    prb_f = open('%s_probe.pdb'%outpath,'w')
    if bildpath == None:
        bild_axis = 'x_axis.bild'
        bild_bond = 'x_bond.bild'
        bild_else = 'x_else.bild'
        bild_intra = 'x_intra.bild'
        bild_polar = 'x_polar.bild'
        bild_probe = 'x_probe.bild'
    else:
        bild_axis = bildpath['bild_axis']
        bild_bond = bildpath['bild_bond']
        bild_else = bildpath['bild_else']
        bild_intra = bildpath['bild_intra']
        bild_polar = bildpath['bild_polar']
        bild_probe = bildpath['bild_probe']
        
    bild_axis_f = open(bild_axis,'w')
    bild_bond_f = open(bild_bond,'w')
    bild_else_f = open(bild_else,'w')
    bild_intra_f = open(bild_intra,'w')
    bild_polar_f = open(bild_polar,'w')
    bild_probe_f = open(bild_probe,'w')

    pos_list_torch = pdb_dict['pos_list']
    resno_list = pdb_dict['resno_list'].tolist()
    resname_list = pdb_dict['resname_list']
    atmname_list = pdb_dict['atmname_list']
    pos_list = pdb_dict['pos_list'].tolist()
    neigh_water_diff = pdb_dict['neigh_water_diff']
    n_water_int_list = pdb_dict['n_water_int_list'].tolist()
    axis_list = pdb_dict['axis_list']
    polar_mask_list = pdb_dict['polar_mask_list']
    grid_diff = pdb_dict['grid_diff']
    polar_vec_list = pdb_dict['polar_vec_list']
    probe_mask_list = pdb_dict['probe_mask_list']
    n_atm = len(pos_list)
    # save position
    atmno = 0
    for i in range(n_atm):
        atmno = i
        pos   = pos_list[i]
        resno   = resno_list[i]
        resname = resname_list[i]
        atmname = atmname_list[i]
        txt = 'ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f\n'%(atmno,atmname, resname, resno,*pos)
        pdb_f.write(txt)

    # save water
    for i in range(n_atm):
        resno   = resno_list[i]
        for j in range(neigh_water_diff.shape[1]):
            if n_water_int_list[i][j] == 1:
                atmno += 1
                
                pos = (neigh_water_diff[i][j] + pos_list_torch[i]).tolist()
                #print(i, j, resno_list[i], resname_list[i], atmname_list[i], pos_list_torch[i], neigh_water_diff[i][j], pos)
                txt = 'HETATM%5d  O   TRU W%4d    %8.3f%8.3f%8.3f\n'%(atmno%100000,(2000+resno),*pos)
                pdb_f.write(txt)

    # save axis
    ax_color = ['red','green','blue']
    for ax in range(3):
        bild_axis_f.write('.color %s\n'%ax_color[ax])
        for i in range(n_atm):
            if polar_mask_list[i] == 1:
                vec_start = pos_list_torch[i].tolist()
                vec_end   = (2.0*axis_list[i][ax] + pos_list_torch[i]).tolist()
                bild_axis_f.write('.arrow %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n'%(*vec_start, *vec_end))


    # save polar_vec
    bild_axis_f.write('.color 22\n') #purple color
    for i in range(n_atm):
        if polar_mask_list[i] == 1:
            vec_start = pos_list_torch[i].tolist()
            vec_end   = (4.5*polar_vec_list[i] + pos_list_torch[i]).tolist()
            bild_axis_f.write('.arrow %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n'%(*vec_start, *vec_end))

    #save grid pt
    bild_probe_f.write('.color 5\n') #teal color
    prb_idx = 0
    for i in range(n_atm):
        if polar_mask_list[i] != 1:
            continue
        if probe_mask_list[i] < 1.0:
            continue
        
        for j in range(grid_diff.shape[1]):
            prb_idx += 1
            resno = pdb_dict['resno_list'][i]
            #vec =  (pos_list_torch[i] + grid_diff[i][j]).tolist()   
            vec =  (pos_list_torch[i]).tolist()   
            bild_probe_f.write('.sphere %8.3f %8.3f %8.3f 0.3\n'%(*vec,))
            txt = 'HETATM%5d  N   PRB A%4d    %8.3f%8.3f%8.3f\n'%(prb_idx%100000,resno,*vec)
            prb_f.write(txt)

    #save graph
    if dgl_graph != None:
        edge_intra = [] #white .color white
        edge_bond  = [] #yellow .color 44
        edge_polar = [] #sky blue .color 10
        edge_else  = []

        edge_feature = dgl_graph.edata["edge_feat_0"].detach().cpu().numpy() #[bond,intra,polar]

        edge_src_torch, edge_dst_torch = dgl_graph.edges() 
        edge_src_np = edge_src_torch.detach().cpu().numpy()
        edge_dst_np = edge_dst_torch.detach().cpu().numpy() 

        for i in range(edge_feature.shape[0]):
            if edge_src_np[i] >= n_atm or  edge_dst_np[i] >= n_atm:
                print('not supported edge - index: ',edge_src_np[i],' - ',edge_dst_np[i])
                continue
            vec_start = pos_list[edge_src_np[i]]
            vec_end   = pos_list[edge_dst_np[i]]
            #priority: polar > bond > intra
            #if edge_feature[i][2] == 1:
            #    edge_polar.append( (vec_start,vec_end) )
            #elif edge_feature[i][0] == 1:
            #    edge_bond.append( (vec_start,vec_end) )
            #elif edge_feature[i][1] == 1:
            #    edge_intra.append( (vec_start,vec_end) )
            #else:
            #    edge_else.append( (vec_start,vec_end) )

            #bond removed
            if edge_feature[i][1] == 1:
                edge_polar.append( (vec_start,vec_end) )
            elif edge_feature[i][0] == 1:
                edge_intra.append( (vec_start,vec_end) )
            else:
                edge_else.append( (vec_start,vec_end) )
        
        bild_intra_f.write('.color white\n')
        for edge in edge_intra:
            bild_intra_f.write('.arrow %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n'%(*edge[0], *edge[1]))

        bild_bond_f.write('.color 44\n')
        for edge in edge_bond:
            bild_bond_f.write('.arrow %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n'%(*edge[0], *edge[1]))

        bild_polar_f.write('.color 10\n')
        for edge in edge_polar:
            bild_polar_f.write('.arrow %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n'%(*edge[0], *edge[1]))

        bild_else_f.write('.color red\n')
        for edge in edge_else:
            bild_else_f.write('.arrow %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n'%(*edge[0], *edge[1]))
    pdb_f.close()
    prb_f.close()
    bild_axis_f.close()
    bild_bond_f.close()
    bild_else_f.close()
    bild_intra_f.close()
    bild_polar_f.close()
    bild_probe_f.close()


def pdb_dict_as_pdb (pdb_dict, outpath='x.pdb'):
    f = open(outpath,'w')
    pos_list_torch = pdb_dict['pos_list']
    resno_list = pdb_dict['resno_list'].tolist()
    resname_list = pdb_dict['resname_list']
    atmname_list = pdb_dict['atmname_list']
    pos_list = pdb_dict['pos_list'].tolist()
    neigh_water_diff = pdb_dict['neigh_water_diff']
    n_water_int_list = pdb_dict['n_water_int_list'].tolist()
    axis_list = pdb_dict['axis_list']
    polar_mask_list = pdb_dict['polar_mask_list']
    grid_diff = pdb_dict['grid_diff']
    n_atm = len(pos_list)
    # save position
    atmno = 0
    for i in range(n_atm):
        atmno = i
        pos   = pos_list[i]
        resno   = resno_list[i]
        resname = resname_list[i]
        atmname = atmname_list[i]
        txt = 'ATOM  %5d %4s %3s A%4d    %8.3f%8.3f%8.3f\n'%(atmno,atmname, resname, resno,*pos)
        f.write(txt)

    # save water
    for i in range(n_atm):
        resno   = resno_list[i]
        for j in range(neigh_water_diff.shape[1]):
            if n_water_int_list[i][j] == 1:
                atmno += 1
                pos = (neigh_water_diff[i][j] + pos_list_torch[i]).tolist()
                txt = 'HETATM%5d  O   TRU W%4d    %8.3f%8.3f%8.3f\n'%(atmno%100000,(2000+resno),*pos)
                f.write(txt)

    # save axis
    for i in range(n_atm):
        resno   = resno_list[i]
        if polar_mask_list[i] == 1:
            atmno += 3
            a0 = (1.0*axis_list[i][0] + pos_list_torch[i]).tolist()
            a1 = (1.0*axis_list[i][1] + pos_list_torch[i]).tolist()
            a2 = (1.0*axis_list[i][2] + pos_list_torch[i]).tolist()
            txt0 = 'HETATM%5d  C   AX0 X%4d    %8.3f%8.3f%8.3f\n'%((atmno-2)%100000,(4000+resno),*a0)
            txt1 = 'HETATM%5d  N   AX1 Y%4d    %8.3f%8.3f%8.3f\n'%((atmno-1)%100000,(4000+resno),*a1)
            txt2 = 'HETATM%5d  O   AX2 Z%4d    %8.3f%8.3f%8.3f\n'%((atmno-3)%100000,(4000+resno),*a2)
            f.write(txt0)
            f.write(txt1)
            f.write(txt2)
                
    f.close()