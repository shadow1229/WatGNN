def read_dat(fp):
    result = {}
    f = open(fp,'r')
    lines = f.readlines()
    for line in lines:
        lsp = line.split()
        result[lsp[0]] = lsp[-1]
    f.close()
    return result
def remove(d_dat,d_templ):
    result ={}
    for k in d_templ.keys():
        result[k] = d_dat[k]
    return result
def match(d_res,d_time,outf='x.txt'):
    out = open(outf,'w')
    for t_k in d_time.keys():
        res  = int(d_res[t_k])
        time = float(d_time[t_k])
        out.write("%6d %9.3f\n"%(res,time))
    out.close()


res    = read_dat('nres.txt')
rism = read_dat('3drism_time.dat')
foldx  = read_dat('foldx_time.dat')
wkgb   = read_dat('wkgb_time.dat')
cnn   = read_dat('cnn_time.dat')
cnn_cpu   = read_dat('cnn_cpu_time.dat')
gnn    = read_dat('gnn_time.dat')

rism2  = remove(rism,foldx)
foldx2 = remove(foldx,foldx)
wkgb2  = remove(wkgb,foldx)
cnn2   = remove(cnn,foldx)
cnn_cpu2   = remove(cnn_cpu,foldx)
gnn    = remove(gnn,foldx)

match(res,rism2,'rism_rest.dat')
match(res,foldx2,'foldx_rest.dat')
match(res,wkgb2,'wkgb_rest.dat')
match(res,cnn2,'cnn_rest.dat')
match(res,cnn_cpu2,'cnn_cpu_rest.dat')
match(res,gnn,'gnn_rest.dat')
