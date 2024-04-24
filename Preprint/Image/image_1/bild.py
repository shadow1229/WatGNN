import os,sys
import numpy as np
def read_pdb_as_vecs(fpath):
    result = []
    f = open(fpath,'r')
    lines = f.readlines()
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            result.append([line[30:38], line[38:46], line[46:54]])
    f.close()
    return result

def read_bild(fpath):
    result = []
    color = ''
    f = open(fpath,'r')
    lines = f.readlines()
    for line in lines:
        lsp = line.strip().split()
        if line.startswith('.color'):
            color = lsp[1]
        elif line.startswith('.arrow'):
            v0 = lsp[1:4]
            v1 = lsp[4:7]
            data = {'color':color, 'type':'.arrow', 'v0':v0 , 'v1':v1}
            result.append(data)
        elif line.startswith('.sphere'):
            v0 = lsp[1:4]
            data = {'color':color, 'type':'.sphere', 'v0':v0 }
            result.append(data)
    f.close()
    return result

def is_same_vector(v0,v1):
    v0_arr  =  np.array([float(x.strip()) for x in v0])
    v1_arr  =  np.array([float(x.strip()) for x in v1])
    if np.linalg.norm(v1_arr - v0_arr) < 0.01:
        return True
    else:
        return False
    
def filter_bild(vecs, bild):
    #vecs / bild: bumber as string
    result = []
    for b in bild:
        is_exist = False
        if 'v0' in b.keys():
            v0 = b['v0']
            for v_vec in vecs:
                #print(v_vec, v0)
                if is_same_vector(v0,v_vec):
                    is_exist = True
                    break
        if (not is_exist) and 'v1' in b.keys():
            v1 = b['v1']
            for v_vec in vecs:
                if is_same_vector(v1,v_vec):
                    is_exist = True
                    break
        if is_exist:
            result.append(b)
    return result

def filter_bild_and(vecs, bild):
    #vecs / bild: bumber as string
    result = []
    for b in bild:
        is_exist = False
        if 'v0' in b.keys():
            v0 = b['v0']
            for v_vec in vecs:
                #print(v_vec, v0)
                if is_same_vector(v0,v_vec):
                    is_exist = True
                    break
        if (is_exist) and 'v1' in b.keys():
            is_exist = False #to check v1 exists in vecs as well
            v1 = b['v1']
            for v_vec in vecs:
                if is_same_vector(v1,v_vec):
                    is_exist = True
                    break
        if is_exist:
            result.append(b)
    return result

def write_bild(bild, fpath):
    if len(bild) > 0:
        color = bild[0]['color']
    else:
        color = 'red'
    f = open(fpath,'w')
    f.write('.color %s\n'%color)

    for b in bild:

        if b['type'] == '.arrow':
            f.write('%s %8s %8s %8s %8s %8s %8s 0.03\n'%('.cylinder',*b['v0'],*b['v1']))
        elif b['type'] == '.sphere':
            f.write('%s %8s %8s %8s 0.3\n'%(b['type'],*b['v0']))
    f.close()

vecs = read_pdb_as_vecs('probe.pdb')
bild = read_bild('5IAI_A_else.bild')
bild_filt = filter_bild(vecs, bild)
write_bild(bild_filt, '5IAI_A_else_filt.bild')    

vecs = read_pdb_as_vecs('5IAI_A_pep.pdb')
bild = read_bild('5IAI_A_bond.bild')
bild_filt = filter_bild(vecs, bild)
write_bild(bild_filt, '5IAI_A_bond_filt.bild')    

bild = read_bild('5IAI_A_intra.bild')
bild_filt = filter_bild(vecs, bild)
write_bild(bild_filt, '5IAI_A_intra_filt.bild')    

bild = read_bild('5IAI_A_polar.bild')
bild_filt = filter_bild_and(vecs, bild)
write_bild(bild_filt, '5IAI_A_polar_filt.bild')    