import math
import os,sys,copy,shutil
import numpy as np
from scipy.spatial.distance import cdist
MAX_DIST = 10.0
#=============================================================
#generate ligand pocket water
def getwater(fpath):
    result_tmp = []
    result_lines = []
    f = open(fpath,'r')
    lines = f.readlines()
    for line in lines:
        if not line.startswith('HETATM'):
            continue
        if not line[17:20] == 'HOH':
            continue
        result_lines.append(line)
        vec = [float(line[30+8*i:38+8*i]) for i in range(3)]
        result_tmp.append(vec)

    result = np.array(result_tmp)
    return result_lines, result

def getmetal(fpath):
    result_tmp = []
    result_lines = []
    f = open(fpath,'r')
    lines = f.readlines()
    for line in lines:
        if not line.startswith('HETATM'):
            continue
        
        atmtype = line[76:78].lstrip().upper()
        if atmtype == '':
            atmtype_tmp = line[12:14].strip() 
            atmtype = ''.join( [ i for i in atmtype_tmp if not i.isdigit() ] )
        
        if atmtype not in ['CA','MG','ZN','HG','MN', \
                         'CO','FE','NI','CU','CD']:
            continue
        result_lines.append(line)
        vec = [float(line[30+8*i:38+8*i]) for i in range(3)]
        result_tmp.append(vec)

    result = np.array(result_tmp)
    return result_lines, result

def get_sybyl_lig(atmtype):
    #change atmtype in mol2 into drugscore form
   
     
    ds_list = ['Br'   , 'C.2', 'C.3' , 'C.ar' , 
               'C.cat', 'Cl' , 'F'   , 'I'    , 
               'Met'  , 'N.3', 'N.am', 'N.ar' , 
               'N.pl3', 'O.2', 'O.3' , 'O.co2', 
               'P.3'  , 'S.3']
    if atmtype in ds_list:
        return ds_list.index(atmtype)
    if atmtype == 'O.w':
        return ds_list.index('O.3')
    if atmtype == 'N.2':
        return ds_list.index('N.ar')
    if atmtype == 'N.4':
        return ds_list.index('N.3')
    if atmtype == 'S.2':
        return ds_list.index('S.3')
    if atmtype in ['Co','Ca','Mg','Zn','Hg','Mn', \
                     'Co.oh','Fe','Ni','Cu','Cd']:
        return ds_list.index('Met')
    return -1

def getligand(fpath):
    result_tmp = []
    f = open(fpath,'r')
    lines = f.readlines()
    tp = None
    for line in lines:
        if line.startswith('@'):
            tp = line.strip().lstrip('@<TRIPOS>')
            continue
        if tp == 'ATOM':
            lsp = line.split()
            if len(lsp) != 9:
                print (line)
                raise ValueError
            
            atmtype = lsp[5]
            if atmtype.startswith('H'):
                continue 
            if atmtype.startswith('C'):
                if not atmtype.startswith('Cl'):
                    continue 
                pass
            vec = [float(lsp[2+i]) for i in range(3)]
            result_tmp.append(vec)

    result = np.array(result_tmp)
    return result

def getligand2(fpath):
    vecs   = []
    idxs   = []
    types  = []
    n_bnd  = []
    bvecs_sum  = []
    bvecs  = []
    n_lig = 0
    f = open(fpath,'r')
    lines = f.readlines()
    tp = None
    for line in lines:
        if line.startswith('@'):
            tp = line.strip().lstrip('@<TRIPOS>')
            continue
        if tp == 'ATOM':
            lsp = line.split()
            if len(lsp) != 9:
                print (line)
                raise ValueError
            
            idx     = int(lsp[0])
            atmname = lsp[5]
            if atmname.startswith('H'):
                continue 
            vec = np.array([float(lsp[2+i]) for i in range(3)])
            sybyl_idx  = get_sybyl_lig(atmname)

            #if atmtype.startswith('C'):
            #    if not atmtype.startswith('Cl'):
            #        continue 
            #    pass
            n_lig += 1
            vec = np.array([float(lsp[2+i]) for i in range(3)])
            vecs.append(vec)
            idxs.append(idx)
            types.append(sybyl_idx)
            n_bnd.append(0)
            bvecs_sum.append(np.zeros(3))
        elif tp == 'BOND':
            #1    2    1 1  (bnd_idx, atm_i, atm_j, bnd_typ)
            lsp = line.split()
            atm_i = int(lsp[1])
            atm_j = int(lsp[2])
           
            if (atm_i not in idxs) or (atm_j not in idxs):
                continue 
            idx_i = idxs.index(atm_i)
            idx_j = idxs.index(atm_j)
            n_bnd[idx_i] += 1
            n_bnd[idx_j] += 1
            bvecs_sum[idx_i] += vecs[idx_j]
            bvecs_sum[idx_j] += vecs[idx_i]

    for i in range(len(idxs)):
        if n_bnd == 0:
            bvecs.append(np.zeros(3))
        else: 
            bvecs.append(bvecs_sum[i]/float(n_bnd[i]))

    vecs_out   = []
    types_out  = []
    n_bnd_out  = []
    bvecs_out  = []

    for i in range(len(idxs)):
        vecs_out.append(vecs[i]) 
        types_out.append(types[i]) 
        n_bnd_out.append(n_bnd[i]) 
        bvecs_out.append(bvecs[i]) 

    result = {'vecs':np.array(vecs_out),
              'types':types_out,
              'n_bnd':n_bnd_out,
              'bvecs':np.array(bvecs_out),
              'n_lig':n_lig}

    return result

def get_lig_pckt(in_path,out_path_prefix,pdbpath,ligpath,ang_cut = 100.,dist_cut=4.0):
    ds_list = ['all']
    cos_cut = np.cos(ang_cut * np.pi / 180.)
    #prints water having h-bonding N/O ligand atom -> all
    #index = open('INDEX_refined_set.2019','r')
    

    water_out_paths = ['%s_%s.pdb'%(out_path_prefix,ds_list[i]) for i in range(len(ds_list))]
    water_out = [open(water_out_paths[i],'w') for i in range(len(ds_list))]
    water_lines, water = getwater(in_path)
    metal_lines, metal = getmetal(pdbpath)
    ligand_pck = getligand2(ligpath)
    #ligand_pck = {'vecs':np.array(vecs_out),
    #              'types':types_out,
    #              'n_bnd':n_bnd_out,
    #              'bvecs':np.array(bvecs_out)
    #              'n_lig':n_lig }
    ligand = ligand_pck['vecs']
    ligand_types = ligand_pck['types']
    ligand_n_bnd = ligand_pck['n_bnd']
    ligand_bvecs = ligand_pck['bvecs']
    ligand_n_heavy = ligand_pck['n_lig']
    print(len(ligand), len(metal))
    if (len(ligand) + len(metal)) == 0 or len(water) == 0:
        print ('%s %10d %10d %10d'%(in_path,0,0,ligand_n_heavy))
    else:
        #deal with metal
        if len(metal) >=1:
            dist = cdist(water,metal)
            n_interact_wat = 0
            n_wat = len(dist)
            is_bnd = [0 for i in range(n_wat)]
            for i,w in enumerate(dist): #i: index of water
                for j in range(len(w)):
                    if w[j] < dist_cut:
                        tp = ds_list.index('Met')
                        water_out[tp].write(water_lines[i])
                        water_out[-1].write(water_lines[i])
                        break

        if len(ligand) >=1:
            dist = cdist(water,ligand)
            n_interact_wat = 0
            n_wat = len(dist)
            is_bnd = [0 for i in range(n_wat)]
            for i,w in enumerate(dist): #i: index of water
                for j in range(len(w)):
                    if w[j] < dist_cut:
                        tp = ligand_types[j]
                        #18
                        if tp != -1:
                            water_out[tp].write(water_lines[i])
                        water_out[-1].write(water_lines[i])
                        break

        for i in range(len(ds_list)):
            water_out[i].close()
        del(water_out)




#=============================================================
def read_n_interf(fpath='n_interf.txt'):
    f = open(fpath,'r')
    lines = f.readlines()
    result = {}
    for line in lines:
        if line.startswith('#'):
            continue
        lsp = line.strip().split()
        result[lsp[0]] = int(lsp[1])
    return result

def map_water(refw, modw, n_pred=[]):
    if len(n_pred) == 0:
        n_pred = [len(modw)]
    #n_water = min(len(refw), len(modw), min(n_pred))
    n_water =len(refw)
    #print(len(modw), len(refw))
    if len(modw) == 0:
        pair_s = []
        for n in n_pred:
            pair_s.append([MAX_DIST for i in range(n_water)])
        return np.array(pair_s)

    dist0 = cdist(modw, refw)
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
                k = np.unravel_index(np.argmin(dist), dist.shape) #k: index of minimum dist from dist[:n]
                pair.append(dist[k])
                dist = np.delete(dist, k[0], 0)
                dist = np.delete(dist, k[1], 1)
        pair_s.append(pair)
    return np.array(pair_s)

def sort_list(vals,idxs):
    zp = zip(idxs,vals)
    z  = [x for idxs, x in sorted(zp)]
    return z

def read_pdb(pdb_fn,cutoff_min=-90.0,cutoff_max=999.0):
    water        = []
    scorelist_r  = []#reverse
    scorelist    = []#reverse
    with open(pdb_fn) as fp:
        for line in fp:
            if line.startswith("HETATM"):
                if line[17:20] not in ['WAT','HOH']:
                    continue
                if line[12:16].strip() != 'O':
                    continue
                #bugfix - 210322
                #water.append((line[30:38], line[38:46], line[46:54]))
                try:
                    score = float(line[60:66])
                    if (score <cutoff_max) and (score>cutoff_min):
                        water.append((line[30:38], line[38:46], line[46:54]))
                        scorelist.append(score)
                        scorelist_r.append(-1.0*score)
                except:
                    print ("%s - No score"%pdb_fn)
                    water.append((line[30:38], line[38:46], line[46:54]))
                    scorelist.append(None)
                    scorelist_r.append(None)
    is_avail = True
    for score in scorelist_r:
        if score == None:
            is_avail = False
            break

    if is_avail:
        water_sorted = sort_list(water,scorelist_r) #to sort with desc order.
        #print (scorelist)
        scorelist.sort()
        scorelist.reverse()
        water   = np.array(water_sorted,   dtype=float)

    return water,scorelist

def get_idxs(fpath):
    #batch:4
    result = []
    with open(fpath,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith('#'):
                continue
            lsp = line.split() 
            idx = lsp[0]
            result.append(idx)
    return result
def get_npred(cuts,scores):
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
def get_npred_n(n_cryst, cutv_list, n_pred):
    result = [min(int(cutv*n_cryst) , n_pred) for cutv in cutv_list]
    return result
def get_npred_ires(n_cryst, cutv_list, n_pred):
    result = [min(int(cutv*n_cryst) , n_pred) for cutv in cutv_list]
    return result
def run(env):
    cutoff_list = env['cutoff_list']
    cutv_list   = env['cutv_list']
    cutv_mode   = env['cutv_mode']

    ans_matrix  = env['ans']
    path_matrix = env['prd']

    idxs        = env['idxs']
    train       = env['train']
    prefix      = env['prefix']
    log_path         = env['log']
    if train:
        tt = 'train'
    else:
        tt = 'test'

    if cutv_mode not in ['res','ncryst','score']:
        raise ValueError
    

    
    nowater = [] 
    #excluded =['3nik','4fxq','4riu','6g14']
    excluded = []
    logf = open(log_path,'w')

    dict_ires = None
    if cutv_mode == 'res':
        dict_ires = read_n_interf(fpath='n_interf.txt')

    n_cryst_sum      = 0
    n_ires_sum      = 0
    n_pred_sum      = [0 for i in range(len(cutv_list))] 
    hit_sum      = [ [0 for i in range(len(cutv_list))] for j in range(len(cutoff_list))]
    acc_sum      = [ [0 for i in range(len(cutv_list))] for j in range(len(cutoff_list))]
    cov_sum      = [ [0 for i in range(len(cutv_list))] for j in range(len(cutoff_list))]
    rmsd_sum   = [0.0 for i in range(len(cutv_list))]
    prop_sum   = [0.0 for i in range(len(cutv_list))]
    n_trg =  0.0

    logf.write("%s %s %8s %8s %8s %8s %8s"%('#trg', 'mode', 'n_pred', 'n_cryst','ires', 'prop', 'rmsd' ))
    for cutoff in cutoff_list:
        logf.write(" cov%3.1f"%cutoff)
    for cutoff in cutoff_list:
        logf.write(" acc%3.1f"%cutoff)
    logf.write('\n')
    for id0_ind, id0 in enumerate(idxs):

        #print (id0)
        #cryst_path ='vec_result/%s/%s/%s_cov.pdb'%(tt,ans_dir,id0)
        #cryst_path ='pdb/%s/%s.pdb'%(ans_dir,id0)
        if cutv_mode == 'res':
            ires = dict_ires[id0]
        else:
            ires = -1
        cryst_path = ans_matrix[id0_ind]
        #debug - 210204    
        if id0 in excluded:
            continue
        cryst,score_cryst =  read_pdb(cryst_path,cutoff_max=40.0) #list of vectors
        n_cryst = len(cryst)
        if len(cryst) == 0:
            if id0 not in excluded:
                excluded.append(id0)
            nowater.append('%s_%s'%(tt,id0) )
            #print(tt,id0, 'crystal - no water near ligand')
            continue
        n_trg += 1.0

        pred ,score_pred  =  read_pdb(path_matrix[id0_ind]) #TODO
       
        if cutv_mode == 'res':
            n_pred = get_npred_ires(ires, cutv_list, len(pred))
        elif cutv_mode == 'ncryst':
            n_pred = get_npred_n(len(cryst),cutv_list,len(pred))
        elif cutv_mode == 'score':    
            n_pred = get_npred(cutv_list,score_pred)

        for i in range(len(cutv_list)): 
            n_pred_sum[i] += n_pred[i]
        n_cryst_sum += n_cryst

        dist = map_water(cryst, pred, n_pred=n_pred)


        rmsd_matrix = (np.sum(dist**2, axis=1))
        lt =  [ np.where(dist<cutoff)[0] for cutoff in cutoff_list ]
        #trg cutv n_pred n_wat n_res prop rmsd cov05 cov10 cov15 cov20 acc05 acc10 acc15 acc20 
        hit = [ np.array([  len(np.where(lt[i]==k)[0]) for k in range(len(cutv_list))]) for i in range(len(cutoff_list))]
        #print(hit)
        acc      = [ [0 for i in range(len(cutv_list))] for j in range(len(cutoff_list))]
        cov      = [ [0 for i in range(len(cutv_list))] for j in range(len(cutoff_list))]
        for i in range(len(cutv_list)): 

            if n_pred[i] == 0:
                rmsd = MAX_DIST
            else:
                rmsd   = np.sqrt(rmsd_matrix[i]/float(n_pred[i])) #same with n_predhit
            rmsd_sum[i] += rmsd

            if cutv_mode == 'res':
                prop   =  float(n_pred[i])/(float(ires)) 
            elif cutv_mode == 'ncryst':
                prop   =  float(n_pred[i])/(float(n_cryst)) 
            elif cutv_mode == 'score':    
                prop   =  float(n_pred[i])/(float(n_cryst)) 

            prop_sum[i] += prop
            for j in range(len(cutoff_list)):
                hit_sum[j][i] += hit[j][i]

                if n_pred[i] == 0:
                    acc[j][i] = 0
                else: 
                    acc[j][i]  =  100.0* hit[j][i] / (float(n_pred[i])) #* len(idxs))

                cov[j][i]  =  100.0* hit[j][i] / (float(n_cryst)) # * len(idxs))
                
                acc_sum[j][i] += acc[j][i] 
                cov_sum[j][i] += cov[j][i]
    
            logf.write("%s %s %8d %8d %8d %8.3f %8.3f"%(id0, cutv_mode, n_pred[i], n_cryst,ires, prop, rmsd))
            for j,cutoff in enumerate(cutoff_list):
                logf.write(" %8.3f"%cov[j][i])
            for j,cutoff in enumerate(cutoff_list):
                logf.write(" %8.3f"%acc[j][i])
            logf.write('\n')

    for i in range(len(cutv_list)): 
        logf.write("%s %s %8s %8s %8s %8.3f %8.3f"%('summary', cutv_mode, '-', '-','-', prop_sum[i]/n_trg, rmsd_sum[i]/n_trg))
        for j,cutoff in enumerate(cutoff_list):
            logf.write(" %8.3f"%(cov_sum[j][i]/n_trg))
        for j,cutoff in enumerate(cutoff_list):
            logf.write(" %8.3f"%(acc_sum[j][i]/n_trg))
        logf.write('\n')          

    logf.close()

def get_ligidxs():
    path = 'INDEX_refined_set.2019'
    f = open(path,'r')
    result = []
    lines = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        else:
            lsp = line.split()
            result.append(lsp[0])
    f.close()
    return result

def main(start, num):    
    run_range = range(start,start+num)
    n_dl = 1
    cutoff_list = [0.5,1.0,1.5,2.0]
    testidxs   = get_idxs(fpath='./pdbbind_clean.txt')
    
    cutv_list  = [0.5*i for i in range(1,51)]
    cutv_mode  = "ncryst" 
    pdbbind_dir = './refined-set'    
    prefixs = ['gnn_result']
    ds_list = ['all']
    
    for pref_i, prefix in enumerate(prefixs):
        if pref_i not in run_range:
            continue

        for trg in testidxs:
            in_path = '%s/%s.pdb'%(prefix,trg)
            pdbpath = '%s/%s/%s_protein.pdb'%(pdbbind_dir,trg,trg)
            ligpath = '%s/%s/%s_ligand.mol2'%(pdbbind_dir,trg,trg)
            out_path_prefix = '%s/%s_wat_lig'%(prefix,trg)
            get_lig_pckt(in_path,out_path_prefix,pdbpath,ligpath,ang_cut = 90.,dist_cut=5.0)      

        for i in range(len(ds_list)):
            pp = prefixs[pref_i]
            prd_dir = '%s'%prefixs[pref_i]
            
            prd_train = [ './%s/%s_wat_lig_%s.pdb'%(prd_dir,id0,ds_list[i])  for id0 in trainidxs ]
            prd_test  = [ './%s/%s_wat_lig_%s.pdb'%(prd_dir,id0,ds_list[i])  for id0 in testidxs ] #due to every trg is predicted in train folder
            ans_train = ['%s/%s/%s_wat_lig_%s.pdb'%(pdbbind_dir,id0,id0,ds_list[i]) for id0 in trainidxs]
            ans_test  = ['%s/%s/%s_wat_lig_%s.pdb'%(pdbbind_dir,id0,id0,ds_list[i]) for id0 in testidxs]

            #ans_matrix_test = [ '/home/sonic1229/pdbbind/refined-set/%s/%s_full_wat.pdb'%(id0,id0)  for id0 in testidxs ]
            env_train = { 'cutoff_list':cutoff_list, 
                    'ans':ans_train, 'idxs':trainidxs, 'prd':prd_train,
                    'prefix':'%s_%s'%(prefix,ds_list[i]), 'train':True, "cutv_mode":cutv_mode, "cutv_list":cutv_list,'log':'%s_%s_train.log'%(prefix,ds_list[i])}
            env_test = { 'cutoff_list':cutoff_list, 
                    'ans':ans_test, 'idxs':testidxs, 'prd':prd_test,
                    'prefix':'%s_%s'%(prefix,ds_list[i]), 'train':False, "cutv_mode":cutv_mode, "cutv_list":cutv_list,'log':'%s_%s_test.log'%(prefix,ds_list[i])}
            run(env_train)
            run(env_test)

if __name__ == "__main__":
    try:
        start = int(sys.argv[1])
        num   = int(sys.argv[2])
        main(start,num)
    except:
        start = 0
        num =999999999999
        main(start,num)
