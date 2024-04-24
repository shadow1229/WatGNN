import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

config = {
             'channels_div': 2, 
             'fiber_edge': [(0, 3)], #intra-residue edge + polar-polar edge with 10A cutoff + 3 classification - bond / intra / long-range(or polar-polar)             
             'fiber_hidden': [(0, 64), (1, 64), (2, 64)], 
             'fiber_init': [(0, 16)],
             'fiber_init_pass': [(0, 16)], #fiber-1 was removed due to fiber-1 in the input node feature is removed and initialization model does no interaction with edge, so no fiber-1 could exist in the output of initialization model 
                                           #(this will replace fiber_pass in the output of initialization model and input of interaction model in the ablation model)   
             'fiber_out': [(0, 2 ), (1, 2)], 
             'fiber_pass': [(0, 64), (1, 32)], 
             'fiber_struct': [(0, 64), (1, 32)], 
             'mid_dim':32,
             'loss_weight': {'n_water': 1.0, 'pos_RMSE': 1.0},  
             'lr':0.001,   
             'low_memory': True, 
             'nonlinearity': 'elu', 
             'norm': [True, True], 
             'num_graph_layers': 6,  #6->10
             'num_heads': 8, 
             'num_linear_layers': 6, 
             'clust_radius':2.0, #for clustering in predicted points
             'grid_start':-4.5,  #start point of grid (-4.5A from atom crd)
             'interval':4.5,     #grid interval 
             'water_cutoff':4.5,
             #'score_cutoff':0.5,
             'score_cutoff':0.1,
             'n_grid':2,         #maximum number of grid 
             'device':device,
             'shuffle':True, #enables shuffle for training 
             'crop': True,   #enables cropping for training 
             'crop_radius':15.0, #crop sphere radius 
             'random_rotation':True,#enables random rotation for training
             'n_batch':4, 
             'state_dict_dir':'./network',  

             'debug':False,
             'log_dir':'gnn_log'
}
