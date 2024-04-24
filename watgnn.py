import os, glob, time,copy,random
import numpy as np

from torch import nn,optim
import torch
import pickle
import dgl
from scipy.spatial.distance import cdist
from scipy.stats import special_ortho_group #for random rotation

#https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer
#SE3Transformer/se3_transformer/model

#git clone https://github.com/NVIDIA/DeepLearningExamples
#cd DeepLearningExamples/DGLPyTorch/DrugDiscovery/SE3Transformer

from se3_transformer import Fiber, SE3Transformer
from se3_transformer.layers import LinearSE3, NormSE3

from watgnn_features import bond_dict, polar_vec_dict, aux_vec_dict, charge_dict, hyb_dict 
from watgnn_visualization import pdb_dict_as_pdb, pdb_dict_as_pdb_new
from watgnn_input_preprocess import transform, merge_pdb_dicts, get_probe
from watgnn_input import read_paths,read_pdb_old, read_dataset, read_dataset_simple, read_pdbbind_simple
from watgnn_models import Model
from watgnn_evaluation import eval_dataset
from watgnn_config import config
#=============================================================================
 
def main(mode='train',dataset_path=None): #pred_per_atom -> n_grid
    MAX_NORM = 1.0 #for gradient clipping
    device = config['device']
    log_dir = config['log_dir']
    if not os.access(log_dir,0):
        os.mkdir(log_dir)
    errf = open('%s/error.log'%(log_dir),'a')

    #+residue information -random rotation on axis +small loss function for nonpolar atom (to increase prediction accuracy for sidechain)
    #seems to be information on carbon sidechain atom is not sufficient  

    model = Model(config)
    model = model.to(device) 
    optimizer = optim.Adam(model.parameters(), lr=config['lr']) 
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
    
    grid_start = config['grid_start']  #start point of grid (-4.5A from atom crd)
    interval   = config['interval']     #grid interval
    n_grid     = config['n_grid']         #maximum number of grid

    water_cutoff = config['water_cutoff']
    state_dict_dir = config['state_dict_dir']

    #for training (dummy variable)
    trainset = []
    trainset_lig = None
    trainset_eval =[] 
    trainset_eval_lig = None
    testset = []
    testset_lig = None
        
    dataset     = []
    dataset_lig = []
    dataset_f = open(dataset_path,'r')
    dataset_lines = dataset_f.readlines()
    
    for line in dataset_lines:
        if line.startswith('#'):
            continue
        lsp = line.strip().split()
        if len(lsp) == 1:
            dataset.append(lsp[0])
            dataset_lig.append(None)
        elif len(lsp) == 2:
            dataset.append(lsp[0])
            dataset_lig.append(lsp[1])



    gwgnn_dir = os.path.dirname(__file__)
    curr_dir = os.getcwd()
    os.chdir(gwgnn_dir)
    state_dicts = glob.glob('%s/*.dict'%state_dict_dir)
    state_dicts.sort()
    if len(state_dicts) > 0:
        start_epoch = int(state_dicts[-1].split('/')[-1].split('.')[0].split('_')[-1])
        model.load_state_dict(torch.load(state_dicts[-1]))
        #torchviz - make_dot (show gradient graph)
    else:
        start_epoch = 0
    print('start epoch: ',start_epoch)
    os.chdir(curr_dir)

    if mode == 'train':
        model.train()
        for epoch in range(start_epoch):
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
        for epoch in range(start_epoch,10000):
            #shuffle dataset

            if config['shuffle'] == True:
                idxs = list(range(len(trainset)))
                random.shuffle(idxs)
                trainset_sfl = [ trainset[x] for x in idxs ]
                if trainset_lig == None:
                    trainset_lig_sfl = None
                else:
                    trainset_lig_sfl = [trainset_lig[x] for x in idxs]
                random.shuffle(trainset)
            n_batch = config['n_batch']
            for i in range(len(trainset)//n_batch): #pdbpath_chain in trainset:
                if config['shuffle'] == True:
                    pdbpath_chains = trainset_sfl[n_batch*i:n_batch*(i+1)]
                    if trainset_lig == None:
                        ligpath_chains = None
                    else:
                        ligpath_chains = trainset_lig_sfl[n_batch*i:n_batch*(i+1)]
                else:
                    pdbpath_chains = trainset[n_batch*i:n_batch*(i+1)]
                    if trainset_lig == None:
                        ligpath_chains = None
                    else:
                        ligpath_chains = trainset_lig[n_batch*i:n_batch*(i+1)]
                if ligpath_chains == None:
                    pdb_dict_pres = [read_paths([pdbpath_chains[j]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid)  for j in range(n_batch)]
                    pdb_paths = [pdbpath_chains[j][0]  for j in range(n_batch)]
                    #cropping / random rotation
                    pdb_dicts_tmp = [transform(pdb_dict_pres[j],config=config)  for j in range(n_batch)]
                    pdb_dict = merge_pdb_dicts(pdb_dicts_tmp)      

                else:
                    pdb_dict_pres = [read_paths([pdbpath_chains[j], ligpath_chains[j]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid)  for j in range(n_batch)]
                    pdb_dict_lig  = [read_paths([ligpath_chains[j]],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid)  for j in range(n_batch)]
                    pdb_paths = [pdbpath_chains[j][0]  for j in range(n_batch)]
                    #cropping / random rotation
                    #pdb_dicts_tmp = [transform(pdb_dict_pres[j],config=config)  for j in range(n_batch)]
                    pdb_dicts_tmp = [transform(pdb_dict_pres[j],config=config,crop_pdb_dict=pdb_dict_lig[j])  for j in range(n_batch)]
                    pdb_dict = merge_pdb_dicts(pdb_dicts_tmp)              



                if config['debug']:
                    if not os.access('junk/',0):
                        os.mkdir('junk')

                optimizer.zero_grad()
                is_valid_forward = False
                pred=None
                loss=None
                metric=None
                try:
                    pred, loss,metric = model.forward(pdb_dict, pdb_path=pdb_paths[0], chain=None, save_input=config['debug']) # currently, loss = metric
                    is_valid_forward = True
                except Exception as e:
                    is_valid_forward = False
                    print("============model forward failed (possible alloc error)============")
                    errf.write("============model forward failed (possible alloc error)============\n")
                    errf.write('Error log: %s\n\n'%e)
                    del(pred)
                    del(loss)
                    del(metric)
                    torch.cuda.empty_cache()
                    time.sleep(5) #pause 5 seconds
                if is_valid_forward:
                    try:
                        loss.backward()
                        is_valid = True

                        #NAN gradient prevention
                        for name,param in model.named_parameters():
                            if type(param.grad) != torch.Tensor:
                                continue
                            if not (torch.isfinite(param.grad).all()): 
                                is_valid = False

                        if is_valid:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM) 
                            optimizer.step()
                
                        else:
                            print("============NAN occurred============")
                        print(epoch, 'CSE: ', '%6.2f'%metric[0].detach().cpu().numpy(), 'RMSE: ', '%6.2f'%metric[1].detach().cpu().numpy(), 'angle: ','%6.2f'%(metric[2].detach().cpu().numpy()), 'dist: ', '%6.2f'%(metric[3].detach().cpu().numpy()), *pdb_paths)
                        del(pred)
                        del(loss)
                        del(metric)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print('============unable to train (possible alloc error)=======')
                        errf.write('============unable to train (possible alloc error)=======\n')
                        errf.write('Error log: %s\n\n'%e)
                        torch.cuda.empty_cache() 
                        time.sleep(5) #pause 5 seconds

            #try:
            #    if epoch%5 == 4:
            #        if len(trainset_eval) > 0:
            #            eval_dataset(model, trainset_eval, trainset_eval_lig,  config, log_dir = 'gnn_log', log_path = 'gnn_train_epoch%05d.txt'%(epoch+1), result_pdb_dir = 'gnn_result_ablation',label='train'  )
            #        if len(testset) > 0:
            #            eval_dataset(model, testset, testset_lig, config, log_dir = 'gnn_log', log_path = 'gnn_eval_epoch%05d.txt'%(epoch+1), result_pdb_dir = 'gnn_result_ablation',label='validation'  )
            #except:
            #    print('cannot evaluate')

            if epoch%1 == 0:
                os.chdir(gwgnn_dir)
                if not os.access(state_dict_dir ,0):
                    os.mkdir(state_dict_dir)
                torch.save(model.state_dict(), '%s/epoch_%05d.dict'%(state_dict_dir ,(epoch+1)))
                os.chdir(curr_dir)
            scheduler.step()

    elif mode == 'eval':
        if len(dataset) > 0:
            eval_dataset(model, dataset, dataset_lig, config, log_dir = 'gnn_log', log_path = 'gnn_train_epoch%05d.txt'%(start_epoch+1), result_pdb_dir = 'gnn_result',label='train' )

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("usage: watgnn.py [dataset file path]")
        print("dataset file structure: [pdb file path (relative path)] [mol2 file path (relative path, optional)] ")
    else:
        dataset_path = sys.argv[1] 
        main(mode = 'eval', dataset_path=dataset_path)
