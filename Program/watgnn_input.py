import copy
import numpy as np
import torch
from scipy.spatial.distance import cdist
from watgnn_features import bond_dict, polar_vec_dict, aux_vec_dict, charge_dict, hyb_dict 

def read_mol2(fpath_chain):
    res_dict = {"ALA":0 ,"ARG":1 ,"ASN":2 ,"ASP":3 ,
               "CYS":4 ,"GLN":5 ,"GLU":6 ,"GLY":7 ,
               "HIS":8 ,"ILE":9 ,"LEU":10,"LYS":11,
               "MET":12,"PHE":13,"PRO":14,"SER":15,
               "THR":16,"TRP":17,"TYR":18,"VAL":19,
               "HID":8, "HIE":8, "HIP":8,"MSE":12,"PRB":20,"_ELSE":21}
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    polar_atoms = ['N','O',"P","S","F","CL","BR","I",'CA','MG','ZN','HG','MN','CO','FE','NI','CU','CD','CO.OH']

    ligands ={'entities':[], 
              'resno_list':[],
              'resname_list':[], 
              'pos_list':[],
              'res_list':[],  
              'chain_list':[], 
              'aid_list':[], 
              'atmname_list':[], 
              'atmtype_list':[],
              'polar_mask_list':[]}

    proteins ={'entities':[], 
              'resno_list':[],
              'resname_list':[],  
              'pos_list':[],
              'res_list':[], 
              'chain_list':[], 
              'aid_list':[], 
              'atmname_list':[], 
              'atmtype_list':[],
              'polar_mask_list':[]}
    
    water = {'water_pos_tmp':[],
             'water_pos_other':[]}
    
    if len(fpath_chain) == 2:
        fpath = fpath_chain[0]
        chain_sel = fpath_chain[1]
        #print(fpath,chain_sel)
    else:
        fpath = fpath_chain
        chain_sel = None 
    
    f = open(fpath,'r')
    lines = f.readlines()
    atomlines = []
    tp = None
    for line in lines:
        if line.startswith('@'):
            tp = line.strip().lstrip('@<TRIPOS>')
            continue
        if tp == 'ATOM':
            lsp = line.split()
            if len(lsp) < 6:
                continue
            idx     = int(lsp[0])
            if len(lsp) > 6:
                resno_id = lsp[6]
            else:
                resno_id = '.'
            atomlines.append( (resno_id,lsp))

    atomlines_sorted = sorted(atomlines,key=lambda x:x[0])
    for resno_lsp in atomlines_sorted:
        resno_id = resno_lsp[0]
        lsp = resno_lsp[1]
        atmname = lsp[5]
        atmtype = atmname.split('.')[0].upper()
        if atmname.startswith('H'):
            continue 
        vec = np.array([float(lsp[2+i]) for i in range(3)])

        if atmtype in polar_atoms:
            polar_mask = 1
        else:
            polar_mask = 0
        #processed after get all protein chain
        # - to consider ligand atom with different chain with protein, but makes a complex
        if atmtype in atm_dict:
            aid = atm_dict[atmtype]
        elif atmtype in ['CA','MG','ZN','HG','MN', \
                         'CO','FE','NI','CU','CD','CO.OH']:
            aid = atm_dict["M"]
        elif atmtype in ['F','BR','CL','I']:
            aid = atm_dict["X"]
        elif atmtype in ['LI','NA','K',"RB"]:
            aid = atm_dict["1"]
        else:
            aid = atm_dict["_ELSE"]
        lig_entity = '%s_%s'%(fpath,'#')
        lig_resno_long = '%s_%s_%s_lig'%(fpath,'#',resno_id)
        rid_else = res_dict["_ELSE"]

        ligands['entities'].append(lig_entity)
        ligands['resno_list'].append(lig_resno_long)
        ligands['resname_list'].append("LIG")
        ligands['pos_list'].append(vec)
        ligands['res_list'].append(rid_else)
        ligands['chain_list'].append("#")
        ligands['atmtype_list'].append(atmtype)
        ligands['atmname_list'].append(atmname)
        ligands['aid_list'].append(aid)
        ligands['polar_mask_list'].append(polar_mask)

    return {'ligands':ligands, 'proteins':proteins, 'water':water}

def read_cif(fpath_chain,bcut=40.0):
    #fpath_chain : fpath_chain in read_pdb
    #for fpath_chain in paths:

    #_atom_site.id: mandatory
    """
    data_XXXX
    #
    loop_
    _atom_site.group_PDB (ATOM/HETATM - needed for non-compound ver)
    _atom_site.id        (mandatory)
    _atom_site.type_symbol    (atom type)
    _atom_site.label_atom_id   (atom name - needed for non-compound ver)
    _atom_site.label_alt_id    ( alt name. '.' or 'A' will be accounted.) 
    _atom_site.label_comp_id   (compound name - needed for non-compound ver)
    _atom_site.label_asym_id    (chain name)
    _atom_site.label_entity_id   (chain id -> now non-alphabet can be used)
    _atom_site.label_seq_id      (seqid)
    _atom_site.pdbx_PDB_ins_code (maybe for antibody -> ? is enough)
    _atom_site.Cartn_x            
    _atom_site.Cartn_y 
    _atom_site.Cartn_z 
    _atom_site.occupancy 
    _atom_site.B_iso_or_equiv    
    _atom_site.pdbx_formal_charge 
    _atom_site.auth_seq_id 
    _atom_site.auth_comp_id 
    _atom_site.auth_asym_id 
    _atom_site.auth_atom_id 
    _atom_site.pdbx_PDB_model_num 
    """    
    #ATOM   1    N  N   . ALA A 1 4   ? 22.570 -20.626 -5.602 1.00 65.75  ? 5   ALA A N   1 


    res_dict = {"ALA":0 ,"ARG":1 ,"ASN":2 ,"ASP":3 ,
               "CYS":4 ,"GLN":5 ,"GLU":6 ,"GLY":7 ,
               "HIS":8 ,"ILE":9 ,"LEU":10,"LYS":11,
               "MET":12,"PHE":13,"PRO":14,"SER":15,
               "THR":16,"TRP":17,"TYR":18,"VAL":19,
               "HID":8, "HIE":8, "HIP":8,"MSE":12,"PRB":20,"_ELSE":21}
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    polar_atoms = ['N','O',"P","S","F","CL","BR","I",'CA','MG','ZN','HG','MN','CO','FE','NI','CU','CD','CO.OH']

    items = ['_atom_site.group_PDB', '_atom_site.id', '_atom_site.type_symbol',
             '_atom_site.label_atom_id', '_atom_site.label_alt_id', '_atom_site.label_comp_id',
             '_atom_site.label_asym_id', '_atom_site.label_entity_id', '_atom_site.label_seq_id',
             '_atom_site.pdbx_PDB_ins_code','_atom_site.Cartn_x','_atom_site.Cartn_y',
             '_atom_site.Cartn_z','_atom_site.occupancy','_atom_site.B_iso_or_equiv',
             '_atom_site.pdbx_formal_charge','_atom_site.pdbx_formal_charge','_atom_site.auth_seq_id',
             '_atom_site.auth_comp_id','_atom_site.auth_asym_id','_atom_site.auth_atom_id',
             '_atom_site.pdbx_PDB_model_num']

    ligands ={'entities':[], 
              'resno_list':[],
              'resname_list':[], 
              'pos_list':[],
              'res_list':[],  
              'chain_list':[], 
              'aid_list':[], 
              'atmname_list':[], 
              'atmtype_list':[],
              'polar_mask_list':[]}
    
    proteins ={'entities':[], 
              'resno_list':[],
              'resname_list':[],  
              'pos_list':[],
              'res_list':[], 
              'chain_list':[], 
              'aid_list':[], 
              'atmname_list':[], 
              'atmtype_list':[],
              'polar_mask_list':[]}
    
    water = {'water_pos_tmp':[],
             'water_pos_other':[]}

    loop      = [] 
    isloop    = False
    isatmline = False
    idx_dict = {} 

    if len(fpath_chain) == 2:
        fpath = fpath_chain[0]
        chain_sel = fpath_chain[1]
        #print(fpath,chain_sel)
    else:
        fpath = fpath_chain
        chain_sel = None 
    f = open(fpath,'r')
    lines = f.readlines()

    for line in lines:
        if line.startswith('data'):
            continue
        elif line.startswith('#'):
            continue    
        elif line.startswith('loop_'):
            loop      = []
            isloop    = True
            isatmline = False        
        elif line.startswith('_'):
            loop.append(line.strip())
        else:
            if isloop:
                isloop = False
                if '_atom_site.id' in loop:
                    isatmline = True
                    for i, item in enumerate(items):
                        if item in loop:
                            idx_dict[item] = i
                        else:
                            idx_dict[item] = -1
                            
                    mandatory_items = [ '_atom_site.group_PDB', '_atom_site.id',
                                        '_atom_site.type_symbol','_atom_site.label_alt_id',
                                        '_atom_site.Cartn_x','_atom_site.Cartn_y','_atom_site.Cartn_z',
                                        '_atom_site.label_comp_id','_atom_site.label_atom_id']
                    for item in mandatory_items:
                        if item not in loop:
                            isatmline = False
                            
            if isatmline: #this should not be "elif"!!!
                lsp = line.strip().split()
                

                atmgroup = lsp[ idx_dict['_atom_site.group_PDB']]
                #atmno    = int( lsp[idx_dict['_atom_site.id']] )
                atmtype  = lsp[ idx_dict['_atom_site.type_symbol']]

                altname  = lsp[ idx_dict['_atom_site.label_alt_id']] 
                
                x        = float(lsp[ idx_dict['_atom_site.Cartn_x']])
                y        = float(lsp[ idx_dict['_atom_site.Cartn_y']])
                z        = float(lsp[ idx_dict['_atom_site.Cartn_z']])
                      
                resname  = lsp[ idx_dict['_atom_site.label_comp_id']] 
                atmname  = lsp[ idx_dict['_atom_site.label_atom_id']]            
                
                chain  = lsp[ idx_dict['_atom_site.label_asym_id']] #just in case
                #chainno    = int(lsp[ idx_dict['_atom_site.label_entity_id']]) #just in case
                resno_temp       = lsp[ idx_dict['_atom_site.label_seq_id']] #just in case
                resno_insertion  = lsp[ idx_dict['_atom_site.pdbx_PDB_ins_code']] #just in case
                resno = "%s%s"%(resno_temp,resno_insertion)
                #occupancy= float(lsp[ idx_dict['_atom_site.occupancy']]) #just in case
                if (idx_dict['_atom_site.B_iso_or_equiv']) > -1:
                    bfac     = float(lsp[ idx_dict['_atom_site.B_iso_or_equiv']]) #just in case    
                else:
                    bfac     = 99.99

                if altname not in ['.','A']:
                    continue
                if atmtype == 'H':
                    continue

                ispro = atmgroup.startswith("ATOM")
                if not ispro and resname == "MSE":
                    ispro = True
                iswat =  ( (not ispro) and atmname == " O  " and resname in ["HOH","WAT"])
                
                if resname in res_dict.keys():
                    rid = res_dict[resname]
                else:
                    rid = res_dict["_ELSE"]

                if atmtype in atm_dict:
                    aid = atm_dict[atmtype]
                elif atmtype in ['CA','MG','ZN','HG','MN', \
                                 'CO','FE','NI','CU','CD','CO.OH']:
                    aid = atm_dict["M"]
                elif atmtype in ['F','BR','CL','I']:
                    aid = atm_dict["X"]
                elif atmtype in ['LI','NA','K',"RB"]:
                    aid = atm_dict["1"]
                else:
                    aid = atm_dict["_ELSE"]

                vec = np.array([x,y,z])
                if (not ispro) and (not iswat):
                    if atmtype in polar_atoms:
                        polar_mask = 1
                    else:
                        polar_mask = 0
                    #processed after get all protein chain
                    # - to consider ligand atom with different chain with protein, but makes a complex
                    lig_entity = '%s_%s'%(fpath,chain)
                    lig_resno_long = '%s_%s_%s_lig'%(fpath,chain,resno)
                    rid_else = res_dict["_ELSE"]

                    ligands['entities'].append(lig_entity)
                    ligands['resno_list'].append(lig_resno_long)
                    ligands['resname_list'].append("LIG")
                    ligands['pos_list'].append(vec)
                    ligands['res_list'].append(rid_else)
                    ligands['chain_list'].append(chain)
                    ligands['atmtype_list'].append(atmtype)
                    ligands['atmname_list'].append(atmname)
                    ligands['aid_list'].append(aid)
                    ligands['polar_mask_list'].append(polar_mask)

                elif ispro:       
                    if chain_sel != None and chain != chain_sel:
                        continue
                    entity     = '%s_%s'%(fpath,chain)
                    resno_long = '%s_%s_%s_pro'%(fpath,chain,resno)
                    if atmname == " OXT":
                        atmname = " O  "
                    if resname == "MET" and atmname =='SE  ':
                        atmname = ' SD '

                    if atmtype in polar_atoms: #protein only: ignore ligand for polar_mask
                        if atmtype == 'N' and resname == 'PRO': #no hydrogen bond available
                            polar_mask = 0
                        else:
                            polar_mask = 1
                    else:
                        polar_mask = 0

                    proteins['entities'].append(entity)
                    proteins['resno_list'].append(resno_long)
                    proteins['resname_list'].append(resname)
                    proteins['pos_list'].append(vec)
                    proteins['res_list'].append(rid)
                    proteins['chain_list'].append(chain)
                    proteins['atmtype_list'].append(atmtype)
                    proteins['atmname_list'].append(atmname)
                    proteins['aid_list'].append(aid)
                    proteins['polar_mask_list'].append(polar_mask)
                    #if polar_mask == 1:
                    #    polar_pos.append(vec)
                    #    polar_idx.append(len(pos_list)-1)
                else: #water
                    if bfac < bcut:
                        if chain_sel == None or chain == chain_sel: #contain other chain water??? TODO
                            water['water_pos_tmp'].append(vec)
                        else:
                            water['water_pos_other'].append(vec)
    return {'proteins':proteins, 'ligands':ligands, 'water':water}

def read_pdb(fpath_chain,bcut=40.0):
    #fpath_chain : fpath_chain in read_pdb
    #for fpath_chain in paths:

    res_dict = {"ALA":0 ,"ARG":1 ,"ASN":2 ,"ASP":3 ,
               "CYS":4 ,"GLN":5 ,"GLU":6 ,"GLY":7 ,
               "HIS":8 ,"ILE":9 ,"LEU":10,"LYS":11,
               "MET":12,"PHE":13,"PRO":14,"SER":15,
               "THR":16,"TRP":17,"TYR":18,"VAL":19,
               "HID":8, "HIE":8, "HIP":8,"MSE":12,"PRB":20,"_ELSE":21}
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    polar_atoms = ['N','O',"P","S","F","CL","BR","I",'CA','MG','ZN','HG','MN','CO','FE','NI','CU','CD','CO.OH']

    ligands ={'entities':[], 
              'resno_list':[],
              'resname_list':[], 
              'pos_list':[],
              'res_list':[],  
              'chain_list':[], 
              'aid_list':[], 
              'atmname_list':[], 
              'atmtype_list':[],
              'polar_mask_list':[]}
    
    proteins ={'entities':[], 
              'resno_list':[],
              'resname_list':[],  
              'pos_list':[],
              'res_list':[], 
              'chain_list':[], 
              'aid_list':[], 
              'atmname_list':[], 
              'atmtype_list':[],
              'polar_mask_list':[]}
    
    water = {'water_pos_tmp':[],
             'water_pos_other':[]}


    if len(fpath_chain) == 2:
        fpath = fpath_chain[0]
        chain_sel = fpath_chain[1]
        #print(fpath,chain_sel)
    else:
        fpath = fpath_chain
        chain_sel = None 
    f = open(fpath,'r')
    lines = f.readlines()

    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain = line[21]
            alt   = line[16]
            resname = line[17:20]
            bfac = float(line[60:66])
            if alt not in [' ','A']:
                continue
            if resname in res_dict.keys():
                rid = res_dict[resname]
            else:
                rid = res_dict["_ELSE"]
            resno = line[22:27]
            atmtype = line[12:14].strip()
            if atmtype == 'H' or (len(atmtype)==2 and 'H' in [atmtype[0],atmtype[1]]): #ignore hydrogen
                continue
            if atmtype in atm_dict:
                aid = atm_dict[atmtype]
            elif atmtype in ['CA','MG','ZN','HG','MN', \
                             'CO','FE','NI','CU','CD','CO.OH']:
                aid = atm_dict["M"]
            elif atmtype in ['F','BR','CL','I']:
                aid = atm_dict["X"]
            elif atmtype in ['LI','NA','K',"RB"]:
                aid = atm_dict["1"]
            else:
                aid = atm_dict["_ELSE"]

            atmname = line[12:16]

            vec = np.array( [ float(line[30+8*i:38+8*i]) for i in range(3) ] )

            ispro = line.startswith('ATOM') #True: protein atom / False: ligand/water atom
            if not ispro and resname == "MSE":
                ispro = True
            iswat =  ( (not ispro) and atmname == " O  " and resname in ["HOH","WAT"])
            if (not ispro) and (not iswat):
                if atmtype in polar_atoms:
                    polar_mask = 1
                else:
                    polar_mask = 0
                #processed after get all protein chain
                # - to consider ligand atom with different chain with protein, but makes a complex
                lig_entity = '%s_%s'%(fpath,chain)
                lig_resno_long = '%s_%s_%s_lig'%(fpath,chain,resno)
                rid_else = res_dict["_ELSE"]

                ligands['entities'].append(lig_entity)
                ligands['resno_list'].append(lig_resno_long)
                ligands['resname_list'].append("LIG")
                ligands['pos_list'].append(vec)
                ligands['res_list'].append(rid_else)
                ligands['chain_list'].append(chain)
                ligands['atmtype_list'].append(atmtype)
                ligands['atmname_list'].append(atmname)
                ligands['aid_list'].append(aid)
                ligands['polar_mask_list'].append(polar_mask)

                #lig_pos.append(vec) #now ligands are processed like proteins

            elif ispro:       
                if chain_sel != None and chain != chain_sel:
                    continue
                entity     = '%s_%s'%(fpath,chain)
                resno_long = '%s_%s_%s_pro'%(fpath,chain,resno)
                if atmname == " OXT":
                    atmname = " O  "
                if resname == "MET" and atmname =='SE  ':
                    atmname = ' SD '

                if atmtype in polar_atoms: #protein only: ignore ligand for polar_mask
                    if atmtype == 'N' and resname == 'PRO': #no hydrogen bond available
                        polar_mask = 0
                    else:
                        polar_mask = 1
                else:
                    polar_mask = 0

                proteins['entities'].append(entity)
                proteins['resno_list'].append(resno_long)
                proteins['resname_list'].append(resname)
                proteins['pos_list'].append(vec)
                proteins['res_list'].append(rid)
                proteins['chain_list'].append(chain)
                proteins['atmtype_list'].append(atmtype)
                proteins['atmname_list'].append(atmname)
                proteins['aid_list'].append(aid)
                proteins['polar_mask_list'].append(polar_mask)
                #if polar_mask == 1:
                #    polar_pos.append(vec)
                #    polar_idx.append(len(pos_list)-1)
            else: #water
                if bfac < bcut:
                    if chain_sel == None or chain == chain_sel: #contain other chain water??? TODO
                        water['water_pos_tmp'].append(vec)
                    else:
                        water['water_pos_other'].append(vec)
    return {'proteins':proteins, 'ligands':ligands, 'water':water}


def read_paths(paths,water_cutoff = 4.0,grid_start=-4.5, interval=3.0, n_grid=3, bcut=40 ,is_eval=False):
    #water: bcut: 40 
    # [ same chain with target's chain or 
    #   water from other chain and having neighboring target protein atom within 4.5A ] and
    # do not have any neighboring ligand atom within 4.5A

    #polar mask: 1 for protein O/N atom, except PRO N
      
    #loss mask: becomes 0 when if polar atom has any neighboring water atom that has neighboring ligand atom
    # (criteria: both neighboring criteria has 4.5A cutoff)
    # (to ensure the crystal water existence and its position is dependant to the polar protein atom only) 
    
    def v_size_np (v):
        return np.linalg.norm(v)
    def v_norm_np (v):
        return v / v_size_np(v)
    def get_angle(v1,v2): #0~pi
        v_norm1 = v_norm_np(v1)
        v_norm2 = v_norm_np(v2)
        cos_ang = np.inner(v_norm1,v_norm2)
        result  = np.arccos(cos_ang)
        return result
    def deg_to_rad(val):
        return val* np.pi /180.0
    
    def get_grid_idx(wp_diff, axis, grid_start=-4.5, interval=4.5, n_grid=3):
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

    def get_first_last(id, first_elem_list, elem_list):
        #get first and last atmno/resno in specific residue/entity
        #id: index of residue / entity
        #first_elem_list: list of first atmno/resno.
        #elem_list: list of total atom/redidue
        first_id = first_elem_list[id]
        if id < len(first_elem_list)-1:
            last_id= first_elem_list[id+1]-1
        else:
            last_id = len(elem_list)-1
        return first_id, last_id
    
    #water: new method: pw: dist(p,w) < 3.5A / angle(pw, -polar_vec) > 100 deg (h-bond criteria)
    #                   ww: inside the grid  / have one or more w-w interaction(cutoff 3.5A) / have no p-w bond

    #for protein / ligand
    #applied for each atom 
    #max_neigh_wat = maximum amount of water molecules near each protein atom considered for calculating loss.
    #is_eval: True for evaluation, effect: set every protein polar atom as loss-active

    #entity_list = [] # chain / each ligand
    pos_list   = []
    res_list   = []
    resno_list = []
    resno_list_str = [] #for debug only
    resname_list = []
    atm_list   = []
    atmname_list = []
    #for searching atom in specific residue
    res_firstatm_list = [] #first atom index for each residue
    last_res = "NULL"
    last_ent = "NULL"
    entity_firstres_list = []
    #features
    bond_list = [] #[(atm1,atm2),(atm2,atm1),...]
    polar_vec_list = None #[vector for atom in atm_list]
    charge_list = None #[charge for atom in atm_list]
    hyb_list = None #[hybridization for atom in atm_list]
    #masks
    polar_mask_list = [] #0 for nonpolar(not N/O) 1 for polar (N/O)
    loss_mask_list = []
    #water only
    water_pos_tmp  = []
    water_pos      = []
    water_pos_other = [] #water in other chain. would put into water_pos_tmp if dist(water-polar atom) < 5A 
    angle_cutoff = 100.0
    dist_cutoff = 3.5
    dist_cutoff_ww = 3.5       
    entities = [] #path_chain_subchain #subchains applied for ligand 
    resnos = [] #path_chain_resno 
    res_dict = {"ALA":0 ,"ARG":1 ,"ASN":2 ,"ASP":3 ,
               "CYS":4 ,"GLN":5 ,"GLU":6 ,"GLY":7 ,
               "HIS":8 ,"ILE":9 ,"LEU":10,"LYS":11,
               "MET":12,"PHE":13,"PRO":14,"SER":15,
               "THR":16,"TRP":17,"TYR":18,"VAL":19,
               "HID":8, "HIE":8, "HIP":8,"MSE":12,"PRB":20,"_ELSE":21}
    atm_dict = {'C':0,'N':1,'O':2,'S':3,'SE':3,
                "P":4, "M":5, "X":6,"1":7,"PRB":8,"_ELSE":9}
    polar_atoms = ['N','O',"P","S","F","CL","BR","I",'CA','MG','ZN','HG','MN','CO','FE','NI','CU','CD','CO.OH']

    #lig_pos = [] #new 230922
    ligands ={'entities':[], 'resno_list':[], 'pos_list':[], 'chain_list':[], 'aid_list':[], 'atmname_list':[], 'atmtype_list':[],'polar_mask_list':[]}

    polar_pos = [] #new 230922
    polar_idx = [] #new 230922
    #C-N bond length: 1.32 A
    break_cutoff = 2.0
    break_cutoff_sq = break_cutoff**2
    #paths: list of path #not same as GWCNN
    pdb_container = { 'ligands':{'entities':[], 
                                 'resno_list':[],
                                 'resname_list':[], 
                                 'pos_list':[],
                                 'res_list':[],  
                                 'chain_list':[], 
                                 'aid_list':[], 
                                 'atmname_list':[], 
                                 'atmtype_list':[],
                                 'polar_mask_list':[]},
    
                    'proteins':{'entities':[], 
                                'resno_list':[],
                                'resname_list':[],  
                                'pos_list':[],
                                'res_list':[], 
                                'chain_list':[], 
                                'aid_list':[], 
                                'atmname_list':[], 
                                'atmtype_list':[],
                                'polar_mask_list':[]},
    
                    'water':{'water_pos_tmp':[],
                             'water_pos_other':[]}
    }
    for fpath_chain in paths:
        if len(fpath_chain) == 2:
            fpath = fpath_chain[0]
            chain_sel = fpath_chain[1]
        else:
            fpath = fpath_chain
            chain_sel = None
        
        ftype = fpath.split('.')[-1].strip() #pdb / mol2
        if ftype == 'pdb':
            pdb_container_temp = read_pdb(fpath_chain,bcut=bcut)
        elif ftype == 'mol2':
            pdb_container_temp = read_mol2(fpath_chain)
        elif ftype == 'cif':
            pdb_container_temp = read_cif(fpath_chain,bcut=bcut)

        for k in pdb_container_temp.keys():
            for kk in pdb_container_temp[k].keys():
                pdb_container[k][kk].extend(pdb_container_temp[k][kk])

    pro_len = len(pdb_container['proteins']['pos_list'])
    pos_list = pdb_container['proteins']['pos_list']
    res_list = pdb_container['proteins']['res_list']
    resname_list = pdb_container['proteins']['resname_list']
    resno_list_str = pdb_container['proteins']['resno_list']
    polar_mask_list = pdb_container['proteins']['polar_mask_list']
    atmname_list = pdb_container['proteins']['atmname_list']                
    for i in range(pro_len):
        entity = pdb_container['proteins']['entities'][i]
        resno_long = pdb_container['proteins']['resno_list'][i]

        if entity != last_ent:
            last_ent = entity
            entities.append(entity)
            #since this is the first residue in the entity & res_firstatm_list does not updated yet. 
            entity_firstres_list.append(len(res_firstatm_list))

        if resno_long != last_res:
            last_res = resno_long
            resnos.append(resno_long)
            #since this is the first atom for the residue & atm_list does not updated yet. 
            res_firstatm_list.append(len(atm_list))
            
        atm_list.append(pdb_container['proteins']['aid_list'][i])
        resno_id = len(res_firstatm_list) -1
        resno_list.append(resno_id)

        if polar_mask_list[i] == 1:
            polar_pos.append(pdb_container['proteins']['pos_list'][i])
            polar_idx.append(len(atm_list)-1)

    water_pos_tmp   = pdb_container['water']['water_pos_tmp']
    water_pos_other = pdb_container['water']['water_pos_other']


    # get ligands residues have < 4.5A interaction with protein
    if len(pdb_container['ligands']['pos_list']) == 0:
        pass
    elif len(pos_list) == 0 : #add all ligands
        resno_lig = []
        for i in range(len(pdb_container['ligands']['pos_list'])):
            resno_lig.append(pdb_container['ligands']['resno_list'][i])
        
        resno_lig_set = list(set(resno_lig))
        rid_else = res_dict['_ELSE']
        for i in range(len(pdb_container['ligands']['pos_list'])):
            entity     = pdb_container['ligands']['entities'][i]
            resno_long = pdb_container['ligands']['resno_list'][i]
            if resno_long not in resno_lig_set:
                continue

            if entity != last_ent:
                last_ent = entity
                entities.append(entity)
                #since this is the first residue in the entity & res_firstatm_list does not updated yet. 
                entity_firstres_list.append(len(res_firstatm_list))
            if resno_long != last_res:
                last_res = resno_long
                resnos.append(resno_long)
                #since this is the first atom for the residue & atm_list does not updated yet. 
                res_firstatm_list.append(len(atm_list))
            resno_id = len(res_firstatm_list) -1
            pos_list.append(pdb_container['ligands']['pos_list'][i])
            res_list.append(rid_else)

            resname_list.append("LIG")
            resno_list.append(resno_id)
            resno_list_str.append(resno_long)
            atm_list.append(pdb_container['ligands']['aid_list'][i])
            polar_mask = pdb_container['ligands']['polar_mask_list'][i]
            polar_mask_list.append(polar_mask)
            atmname_list.append(pdb_container['ligands']['atmname_list'][i])
            if polar_mask == 1:
                polar_pos.append(pdb_container['ligands']['pos_list'][i])
                polar_idx.append(len(pos_list)-1)
    else:
        dist = cdist(pdb_container['ligands']['pos_list'] , pos_list )
        mindist = [ np.amin(dist[j,:]) for j in range(len(pdb_container['ligands']['pos_list'])) ]
        resno_lig = []
        for i in range(len(pdb_container['ligands']['pos_list'])):
            if mindist[i] < water_cutoff: #4.5A 
                resno_lig.append(pdb_container['ligands']['resno_list'][i])
        
        resno_lig_set = list(set(resno_lig))
        rid_else = res_dict['_ELSE']
        for i in range(len(pdb_container['ligands']['pos_list'])):
            entity     = pdb_container['ligands']['entities'][i]
            resno_long = pdb_container['ligands']['resno_list'][i]
            if resno_long not in resno_lig_set:
                continue

            if entity != last_ent:
                last_ent = entity
                entities.append(entity)
                #since this is the first residue in the entity & res_firstatm_list does not updated yet. 
                entity_firstres_list.append(len(res_firstatm_list))
            if resno_long != last_res:
                last_res = resno_long
                resnos.append(resno_long)
                #since this is the first atom for the residue & atm_list does not updated yet. 
                res_firstatm_list.append(len(atm_list))
            resno_id = len(res_firstatm_list) -1
            pos_list.append(pdb_container['ligands']['pos_list'][i])
            res_list.append(rid_else)

            resname_list.append("LIG")
            resno_list.append(resno_id)
            resno_list_str.append(resno_long)
            atm_list.append(pdb_container['ligands']['aid_list'][i])
            polar_mask = pdb_container['ligands']['polar_mask_list'][i]
            polar_mask_list.append(polar_mask)
            atmname_list.append(pdb_container['ligands']['atmname_list'][i])
            if polar_mask == 1:
                polar_pos.append(pdb_container['ligands']['pos_list'][i])
                polar_idx.append(len(pos_list)-1)

    #append other chain water to water_pos_tmp
    if not( len(polar_pos) == 0 or len(water_pos_other) == 0 ):
        dist = cdist(water_pos_other, polar_pos)
        mindist = [ np.amin(dist[j,:]) for j in range(len(water_pos_other))] 
        for i in range(len(water_pos_other)):
            if mindist[i] < water_cutoff:
                water_pos_tmp.append(water_pos_other[i])

    #loss_mask: 1 when polar atom & far from ligand-affected water molecule & far from edge of training sphere(trasform part) 
    loss_mask_list = copy.deepcopy(polar_mask_list)

    for i in range(len(water_pos_tmp)):
        water_pos.append(water_pos_tmp[i])

    #print(paths[0], len(water_pos_tmp), len(water_pos))
    #add feature
    #bond_list = [] #[(atm1,atm2),(atm2,atm1),...]
    polar_vec_list = np.zeros( (len(pos_list),3) ) #[vector for atom in atm_list]
    axis_list = torch.zeros( (len(pos_list),3,3) , dtype=torch.float32)  #[vector for atom in atm_list]
    charge_list = np.zeros ( (len(pos_list),) )
    hyb_list = np.zeros ( (len(pos_list),) )

    #print(entity_firstres_list)
    for ent_idx in range(len(entity_firstres_list)):
        res_first,res_last = get_first_last(ent_idx, entity_firstres_list,res_firstatm_list)
        
        #print(ent_idx,res_first,res_last)
        for res_idx in range(res_first, res_last+1):
            isfirstres = False #is first residue
            islastres  = False #is last residue
            if res_idx == res_first:
                isfirstres = True
                prev_atm_first,prev_atm_last = None, None
            else:
                prev_atm_first,prev_atm_last = get_first_last(res_idx-1, res_firstatm_list,atm_list)

            if res_idx == res_last:
                islastres = True
                next_atm_first,next_atm_last = None, None
            else:
                next_atm_first,next_atm_last = get_first_last(res_idx+1, res_firstatm_list,atm_list)

            atm_first,atm_last = get_first_last(res_idx, res_firstatm_list,atm_list)
            resname = resname_list[atm_first]
            #atmname_list
            #pos_list
            for atm_idx in range(atm_first,atm_last+1):
                atmname = atmname_list[atm_idx]

                #charge, hybridization============================
                charge =  0
                hyb    = -1
                has_charge = False
                has_hyb    = False
                if resname in charge_dict.keys():
                    if atmname in charge_dict[resname].keys():
                        has_charge = True
                        charge = charge_dict[resname][atmname]

                if resname in hyb_dict.keys():
                    if atmname in hyb_dict[resname].keys():
                        has_hyb = True
                        hyb = hyb_dict[resname][atmname]

                charge_list[atm_idx] = charge
                hyb_list[atm_idx] = hyb
                #if not has_charge:
                #    print('charge: ',resname, atmname, 'not exists')
                #if not has_hyb:
                #    print('HYB: ',resname, atmname, 'not exists')
                #charge, hybridization end========================
                
                #bond  ===========================================
                #N-C bond
                if resname != "LIG":
                    
                    if (not isfirstres) and atmname == " N  ":
                        if prev_atm_first != None:
                            for atm_jdx in range(prev_atm_first,prev_atm_last+1):
                                atmname_other = atmname_list[atm_jdx]
                                if atmname_other == " C  ":
                                    #check chain break
                                    vec_n = pos_list[atm_idx]
                                    vec_c = pos_list[atm_jdx]
                                    if np.sum(np.power((vec_n-vec_c),2)) < break_cutoff_sq:
                                        bond_list.append( (atm_idx,atm_jdx) )  
                                        bond_list.append( (atm_jdx,atm_idx) )  
                    #bonds in same residue
                    for atm_jdx in range(atm_first,atm_last+1):
                        atmname_other = atmname_list[atm_jdx]
                        tup_0 = (atmname, atmname_other)
                        tup_1 = (atmname_other, atmname)
                        if resname in bond_dict.keys():
                            if tup_0 in bond_dict[resname] or tup_1 in bond_dict[resname]:
                                vec_self  = pos_list[atm_idx]
                                vec_other = pos_list[atm_jdx]
                                if np.sum(np.power((vec_self-vec_other),2)) < break_cutoff_sq:
                                    bond_list.append( (atm_idx,atm_jdx) )                                            
                #bond  end========================================

                #polar vec ========================================
                pos_i        = pos_list[atm_idx] #pos_list: list of np.array
                pos_neighs   = []
                aux_vec   = np.zeros(3,)
                polar_vec = np.zeros(3,)

                atmidx_prev_to_next = [0,0]
                
                if isfirstres:
                    atmidx_prev_to_next[0] = atm_first
                else:
                    atmidx_prev_to_next[0] = prev_atm_first

                if islastres:
                    atmidx_prev_to_next[1] = atm_last
                else:
                    atmidx_prev_to_next[1] = next_atm_last

                atm_prev_to_next = pos_list[atmidx_prev_to_next[0]:atmidx_prev_to_next[1]+1]
                n_atm_prev_to_next = len(atm_prev_to_next)
                atm_dist = cdist( [pos_list[atm_idx]], atm_prev_to_next)[0]
                for j in range(n_atm_prev_to_next):
                    atm_jdx = atmidx_prev_to_next[0] + j
                    if atm_dist[j] < break_cutoff:
                        pos_neighs.append(pos_list[atm_jdx])

                for pos_neigh in pos_neighs:
                    polar_vec += (pos_i - pos_neigh)

                norm = np.linalg.norm(polar_vec)
                if norm >= EPS:
                    polar_vec_norm = polar_vec/norm
                else:
                    polar_vec_norm = polar_vec
                polar_vec_list[atm_idx] = polar_vec_norm


                for j in range(atm_first,atm_last+1):
                    aux_vec += (pos_i - pos_list[j])

                norm = np.linalg.norm(aux_vec)
                if norm >= EPS:
                    aux_vec_norm = aux_vec/norm
                else:
                    aux_vec_norm = aux_vec

                if polar_mask_list[atm_idx] == 1:
                    e0 = v_norm_safe(torch.from_numpy(polar_vec_list[atm_idx]), index=0)
                    v1 = torch.from_numpy(aux_vec_norm)
                    u1 = v1 - e0 * inner_product(e0, v1)
                    norm_u1 = torch.norm(u1)
                    if norm_u1 < EPS:
                        ax0 = torch.zeros(e0.size(-1), device=e0.device, dtype=e0.dtype)
                        ax0[0] = 1.0
                        ax1 = torch.zeros(e0.size(-1), device=e0.device, dtype=e0.dtype)
                        ax1[1] = 1.0
                        u1_tmp_0 = torch.cross(e0, ax0)
                        norm_u1_tmp_0 = torch.norm(u1_tmp_0)
                        u1_tmp_1 = torch.cross(e0, ax1)
                        norm_u1_tmp_1 = torch.norm(u1_tmp_1)
                        if norm_u1_tmp_0 > norm_u1_tmp_1:
                            u1 = u1_tmp_0/norm_u1_tmp_0
                        else:
                            u1 = u1_tmp_1/norm_u1_tmp_1
                    e1 = v_norm_safe(u1, index=1) #works
                    e2 = torch.cross(e0, e1) #Nx3 works
                    axis_list[atm_idx][0] = e0
                    axis_list[atm_idx][1] = e1
                    axis_list[atm_idx][2] = e2
                #else: #apply random rotation.... this does not prevent NAN error for using more than 1 axis or using grid point
                #    rot = torch.tensor(special_ortho_group.rvs(3),dtype=torch.float32)
                #    axis_list[atm_idx] = rot


    #make answer - number of water molecules near each grid per atom 
    n_water = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2)), dtype=np.int64) #N -> N x grid^3 #if crystallographic water molecule exists in the grid: 1, else: 0
    n_water_analysis = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2)), dtype=np.int64) #N -> N x grid^3 #for water_analysis, number of crystallographic water molecules in the grid 
    neigh_water_diff = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2),3), dtype=np.float32) #N x max_neigh_wat x 3 -> N x grid^3 x 3

    n_water_ww = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2)), dtype=np.int64) #N -> N x grid^3 #if crystallographic water molecule exists in the grid: 1, else: 0
    n_water_analysis_ww = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2)), dtype=np.int64) #N -> N x grid^3 #for water_analysis, number of crystallographic water molecules in the grid 
    neigh_water_diff_ww = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2),3), dtype=np.float32) #N x max_neigh_wat x 3 -> N x grid^3 x 3
    
    grid_diff = np.zeros((len(pos_list),(n_grid -n_grid//2)*(n_grid**2),3), dtype=np.float32) #N x grid^3 x 3, center of each grid, precalculated for prediction
    #print(len(pos_list), len(water_pos))
    if len(pos_list) == 0 or len(water_pos) == 0:
        dist0 = [ [] ]
        mindist = []
    else:
        dist0 = cdist(pos_list, water_pos)
        #considering only O/N
        for i in range(len(pos_list)): 
            if polar_mask_list[i] != 1:
                dist0[i,:] = 999.999
        mindist = [ np.amin(dist0[:,j]) for j in range(len(water_pos))] #for debugging purpose

    n_wat = len(water_pos)
    n_atm = len(pos_list)
    polar_watidx   = [ None for i in range(n_atm)]
    wat_polaridx = [ [] for i in range(n_wat)] #probe idxs in each water
    wat_watidx   = [ [] for i in range(n_wat)] #water having h-bond with another water (use dist_cutoff)
    if n_wat > 0:
        wwdist = cdist(water_pos, water_pos)
        for i_wat in range(n_wat):
            for j_wat in range(n_wat):
                if i_wat == j_wat:
                    continue
                if wwdist[i_wat][j_wat] < dist_cutoff_ww:
                    wat_watidx[i_wat].append(j_wat)


    for i in range(len(pos_list)):
        if polar_mask_list[i] != 1:
            continue
        e0 = axis_list[i][0]
        e1 = axis_list[i][1]
        e2 = axis_list[i][2]
        #precalculation of grid center
        grid_range = np.arange(n_grid)
        grid_range_x = np.arange(n_grid//2, n_grid)
        gx, gy, gz = np.meshgrid(grid_range_x, grid_range, grid_range, indexing='ij')
        grid_x = (grid_start + interval*(gx+0.5)).reshape(-1)
        grid_y = (grid_start + interval*(gy+0.5)).reshape(-1)
        grid_z = (grid_start + interval*(gz+0.5)).reshape(-1)
        #print(gx,gy,gz)
        #print(grid_x,grid_y,grid_z)
        #print(v_size_np(e0), v_size_np(e1), v_size_np(e2) )

        ex = np.einsum('i,j->ij',grid_x, e0) 
        ey = np.einsum('i,j->ij',grid_y, e1) 
        ez = np.einsum('i,j->ij',grid_z, e2)
        e_val = ex+ey+ez 
        #print(v_size_np(e_val[0]), v_size_np(e_val[1]), v_size_np(e_val[2]), v_size_np(e_val[3]))
        #raise ValueError
        grid_diff[i] = e_val
        #difference vector between atom and water molecule    
        if len(pos_list) == 0 or len(water_pos) == 0:
            continue   
        for j in range(len(dist0[i])):
            d = dist0[i][j]
            if d < water_cutoff:
                v = water_pos[j] - pos_list[i]
                grid_idx = get_grid_idx(v, axis_list[i], grid_start=grid_start, interval=interval, n_grid=n_grid)
                if grid_idx < 0:
                    continue

                hbond_self = has_hbond_wp(v, e0, dist_cutoff=dist_cutoff, angle_cutoff = angle_cutoff)
                if not hbond_self:
                    continue
                
                n_water_analysis[i,grid_idx] += 1

                if n_water[i,grid_idx] == 0:
                    neigh_water_diff[i,grid_idx] = v
                    n_water[i,grid_idx] = 1
                    polar_watidx[i] = j
                else:
                    norm_v = np.linalg.norm(v)
                    norm_prev = np.linalg.norm(neigh_water_diff[i,grid_idx])
                    if norm_v < norm_prev:
                        neigh_water_diff[i,grid_idx] = v
                        polar_watidx[i] = j
                
                #print(v, water_pos[j], pos_list[i], dist_cutoff, angle_cutoff, e0, hbond_self)

    #231025 explicit grid patch (prb; probe) 
    # 1. appending grid positions into pos_list
    # 2. resname list

    #remove non N/O atom
    
    #assign probes to water molecule   
    for i_polar in range(len(polar_watidx)):
        i_wat = polar_watidx[i_polar]
        if i_wat != None:
            wat_polaridx[i_wat].append(i_polar)

    for i in range(n_atm):
        if polar_mask_list[i] != 1:
            continue
        axis = axis_list[i]
        polar_vec = axis[0] #polar_pos - neigh_pos

        for j in range(n_wat):
            #no p-w hbond allowed
            if len(wat_polaridx[j]) > 0:
                continue
            if len(wat_watidx[j]) == 0:
                continue

            v = water_pos[j] - pos_list[i]
            if np.linalg.norm(v,ord=np.inf) > water_cutoff:
                continue

            grid_idx = get_grid_idx(v, axis, grid_start=grid_start, interval=interval, n_grid=n_grid)
            if grid_idx < 0:
                continue

            n_water_analysis_ww[i,grid_idx] += 1

            if n_water_ww[i,grid_idx] == 0:
                neigh_water_diff_ww[i,grid_idx] = v
                n_water_ww[i,grid_idx] = 1
            else:
                norm_v = np.linalg.norm(v)
                norm_prev = np.linalg.norm(neigh_water_diff_ww[i,grid_idx])
                if norm_v < norm_prev:
                    neigh_water_diff_ww[i,grid_idx] = v
    #print(len(pos_list), len(res_list), len(resname_list), len(resno_list), len(atm_list),len(atmname_list),len(polar_vec_list), axis_list.shape)
    #print(len(charge_list), len(hyb_list), len(grid_diff), len(n_water))
    result =  { 'entities':entities, #new
                'entity_firstres_list':entity_firstres_list, #new
                'res_firstatm_list':res_firstatm_list, #new
                'pos_list':torch.from_numpy(np.array(pos_list, dtype=np.float32)),                 #position of atoms
                'res_list':torch.from_numpy(np.array(res_list, dtype=np.int32)), #residue type embedding
                'resname_list':resname_list,
                'resno_list':torch.from_numpy(np.array(resno_list, dtype=np.int32)), #residue number indexing
                'atm_list':torch.from_numpy(np.array(atm_list, dtype=np.int32)), #atom type embedding
                'atmname_list':atmname_list,                   #atom name
                'bond_list':torch.from_numpy(np.array(bond_list, dtype=np.int64)),  #new
                'polar_vec_list':torch.from_numpy(np.array(polar_vec_list, dtype=np.float32)), #polar vector new
                'axis_list':axis_list, #torch tensor.
                'charge_list':torch.from_numpy(np.array(charge_list, dtype=np.float32)), #charge from charmm36 new
                'hyb_list':torch.from_numpy(np.array(hyb_list, dtype=np.float32)), #hybridization, 0 for sp2, 1 for sp3. new
                'n_water_list':torch.from_numpy(np.array(n_water, dtype=np.float32)),              #number of water nearby the atom
                'n_water_int_list':torch.from_numpy(np.array(n_water, dtype=np.int64)),
                'n_water_analysis_int':torch.from_numpy(np.array(n_water_analysis, dtype=np.int64)),
                'grid_diff':torch.from_numpy(np.array(grid_diff, dtype=np.float32)), #center of each grid, precalculated for prediction
                'water_pos':torch.from_numpy(np.array(water_pos, dtype=np.float32)),               #position of total water molecules Nx3
                'water_cutoff':torch.tensor(water_cutoff), #water
                'neigh_water_diff':torch.from_numpy(np.array(neigh_water_diff, dtype=np.float32)), # N x max_neigh x 3, saves difference between water crd and protein atom crd.
                'polar_mask_list':torch.from_numpy(np.array(polar_mask_list, dtype=np.float32)), #polar atom mask
                'loss_mask_list':torch.from_numpy(np.array(loss_mask_list, dtype=np.float32)), #mask for calculating loss.
                'n_water_ww_list':torch.from_numpy(np.array(n_water_ww, dtype=np.float32)),              #number of water nearby the atom
                'n_water_ww_int_list':torch.from_numpy(np.array(n_water_ww, dtype=np.int64)),
                'n_water_analysis_ww_int':torch.from_numpy(np.array(n_water_analysis_ww, dtype=np.int64)),
                'neigh_water_diff_ww':torch.from_numpy(np.array(neigh_water_diff_ww, dtype=np.float32)), # N x max_neigh x 3, saves difference between water crd and protein atom crd.
                'mindist':mindist,
                'resno_list_str':resno_list_str}
    return result


#=============================================================================
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

def read_dataset(dbpath, direc = './pdb', label='train', water_cutoff = 4.5, grid_start=-4.5, interval=3.0, n_grid=3, n_max_trg=None, is_eval=None):
    result = []
    with open(dbpath,'r') as f:
        lines = f.readlines()
        
        pdblist = []
        for line in lines:
            if line.startswith('#'):
                continue
            pdblist.append(line.strip())

        for i, pdbline in enumerate(pdblist):
            if n_max_trg != None:
                if i>=n_max_trg:
                    break
            pdbpath = '%s/%s.pdb'%(direc,pdbline.strip()[:4])
            print(label, pdbpath)

            read_time_start = time.time()
            if is_eval == True:
                pdb_dict = read_pdb([pdbpath],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid, is_eval=is_eval) 
            else:
                pdb_dict = read_pdb([pdbpath],water_cutoff = water_cutoff, grid_start=grid_start, interval=interval, n_grid=n_grid) 
            read_time_end = time.time()
            
            read_time = read_time_end - read_time_start   
            result.append({'path':pdbpath,
                           'pdb_dict':pdb_dict, 
                           'label':label, 
                           'read_time':read_time, 
                           'gnn_time':None,
                           'pos_time':None,
                           'pos_all':None,
                           'pos_filt':None, 
                           'pos_clust':None})
    return result

def read_dataset_simple(dbpath, direc = './pdb',suffix = '.pdb',n_max_trg=None):
    result = []
    with open(dbpath,'r') as f:
        lines = f.readlines()

        pdblist = []
        for line in lines:
            if line.startswith('#'):
                continue
            pdblist.append(line.strip())

        for i, pdbline in enumerate(pdblist):
            if n_max_trg != None:
                if i>=n_max_trg:
                    break

            lsp = pdbline.split(',')
            if len(lsp) == 1:
                pdb = lsp[0].strip()
                chain = None
            elif len(lsp) == 2:
                pdb = lsp[0].strip()
                chain = lsp[1].strip()
            pdbpath = '%s/%s%s'%(direc,pdb,suffix)
            result.append((pdbpath,chain))
    return result

def read_pdbbind_simple(dbpath, direc = './pdb',n_max_trg=None):
    result = []
    result_ligand = []
    with open(dbpath,'r') as f:
        lines = f.readlines()

        pdblist = []
        for line in lines:
            if line.startswith('#'):
                continue
            pdblist.append(line.strip())

        for i, pdbline in enumerate(pdblist):
            if n_max_trg != None:
                if i>=n_max_trg:
                    break

            lsp = pdbline.split(',')
            if len(lsp) == 1:
                pdb = lsp[0].strip()
                chain = None
            elif len(lsp) == 2:
                pdb = lsp[0].strip()
                chain = lsp[1].strip()

            pdbpath = '%s/%s/%s_protein.pdb'%(direc,pdb,pdb)
            mol2path = '%s/%s/%s_ligand.mol2'%(direc,pdb,pdb)
            result.append((pdbpath,chain))
            result_ligand.append((mol2path,chain))
    return result, result_ligand
