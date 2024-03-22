import os,sys
ANGSTROM = "A"

f = open('gnn_result_ablation_150_sorted_all_test.log','r')
lines = f.readlines()
out05 = open('gnn_lig_50_all_c0.5_dlv_00.txt','w')
out10 = open('gnn_lig_50_all_c1.0_dlv_00.txt','w')
out15 = open('gnn_lig_50_all_c1.5_dlv_00.txt','w')
out20 = open('gnn_lig_50_all_c2.0_dlv_00.txt','w')
out05.write('#   prop      acc      cov     rmsd\n')
out10.write('#   prop      acc      cov     rmsd\n')
out15.write('#   prop      acc      cov     rmsd\n')
out20.write('#   prop      acc      cov     rmsd\n')
for line in lines:
    lst = line.strip()
    lsp = lst.split()
    if lsp[0] != 'summary':
        continue
    prop = lsp[5]
    rmsd = lsp[6]
    cov05 = lsp[7]
    cov10 = lsp[8]
    cov15 = lsp[9]
    cov20 = lsp[10]
    
    acc05 = lsp[11]
    acc10 = lsp[12]
    acc15 = lsp[13]
    acc20 = lsp[14]
    out05.write('%8s %8s %8s %8s\n'%(prop,acc05,cov05,rmsd))
    out10.write('%8s %8s %8s %8s\n'%(prop,acc10,cov10,rmsd))
    out15.write('%8s %8s %8s %8s\n'%(prop,acc15,cov15,rmsd))
    out20.write('%8s %8s %8s %8s\n'%(prop,acc20,cov20,rmsd))
    
    

