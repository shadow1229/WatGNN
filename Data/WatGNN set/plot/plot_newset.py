import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib.font_manager import FontProperties #unicode
def read_dat(fpath):
    #wkgb/data(_cmp)/native(relaxed)_summary.dat
    #  N |   RMSD    Ave    Med   f<0.5  f<1.0 f<1.5
    f = open(fpath,'r')
    dump = []
    lines  = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        lsp = line.split()
        x = { 'n'    :int(lsp[0]),
              'RMSD' :float(lsp[2]),
              'cov_05':float(lsp[5]),
              'cov_10':float(lsp[6]),
              'cov_15':float(lsp[7])}
        dump.append(x)

    result_tr = []
    N_cryst = dump[0]['n']
    N_pred = dump[-1]['n']
    for item in dump:
        if item['n'] > N_pred:
            continue

        x = { 'n'    :float(item['n'])/ float(N_cryst),
              'RMSD' :item['RMSD'],
              'cov_05':item['cov_05'],
              'cov_10':item['cov_10'],
              'cov_15':item['cov_15']}
        result_tr.append(x)

    result = {}
    for item in result_tr:
        for k in item.keys():
            if not k in result.keys():
                result[k] = []
            result[k].append(item[k])

    return result
xlabels = ['n']
ylabels = ['RMSD','cov_05','cov_10','cov_15']
#ylabels = ['RMSD','Coverage']
#types_dict = {'train' :'$\mathrm{training}$ $\mathrm{set}$ $\mathrm{(4000}$ $\mathrm{targets)}$',
#              'validation' :'$\mathrm{validation}$ $\mathrm{set}$ $\mathrm{(300}$ $\mathrm{targets)}$',
#              'test' :'$\mathrm{test}$ $\mathrm{set}$ $\mathrm{(3700}$ $\mathrm{targets)}$',
#              }
#        'relaxed':'$\mathrm{perturbed}$ $\mathrm{structure}$ $\mathrm{set}$'}
types_list = ['Training set','Validation set','Test set']
files_list = ['train','validation','test']
#method_dict    ={'wkgb':'GalaxyWater-KGB','3drism':'3D-RISM','foldX':'FoldX',
#                 'GWCNN':'GalaxyWater-CNN','GWGNN':'GWGNN'}
#method_list    = ['GWGNN','GWCNN','wkgb','3drism','foldX']

method_list    = ['gwgnn']

color  = ['#FF0000','#00FF00','#0000FF','#000000','#880088','#008888']
plt.rc('mathtext', fontset='cm')
#plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text',usetex=True)

for method in method_list:
    dat_dict = {}
    for i, f in enumerate(files_list):
        typ   = types_list[i]
        fpath = './%s_%s.dat'%(f,method)
        print (fpath)
        dat = read_dat(fpath)
        dat_dict[typ] = dat  #dat_dict: {'wkgb':{} ,}
        
    for ylabel in ylabels:
        plt.rc('mathtext', fontset='cm')
        #plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
        #plt.rc('text',usetex=True)
        fig = plt.figure(figsize = (6,4),
                         facecolor = 'white',
                         edgecolor = 'black',
                         dpi  = 300
                        )
        prop = FontProperties(size=16) #unicode
        #ax = fig.add_subplot(111)
        ax = fig.add_axes([0.14,0.15,0.80,0.74])
        ax.grid(visible=True, axis='both',linestyle='dotted',color='black')
        ax.legend(bbox_to_anchor=(1.00,1.0))
        title = '$\mathrm{WatGNN}$ $\mathrm{crystal}$ $\mathrm{structure}$ $\mathrm{set}$'
        ax.set_title(r'%s'%title,fontproperties=prop)

        ax.set_xlim(0.0,10.0)
        ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
        if ylabel == 'RMSD':
            ax.set_ylabel(r'$\mathrm{RMSD}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
        elif ylabel == 'cov_05':
            ax.set_ylabel(r'$\mathrm{Coverage}$ $\mathrm{({0.5\AA})}$',fontproperties=prop)
        elif ylabel == 'cov_10':
            ax.set_ylabel(r'$\mathrm{Coverage}$',fontproperties=prop)
        elif ylabel == 'cov_15':
            ax.set_ylabel(r'$\mathrm{Coverage}$ $\mathrm{({1.5\AA})}$',fontproperties=prop)
        else:
            ax.set_ylabel(r'$\mathrm{%s}$'%ylabel,fontproperties=prop)
        ax.set_xticks(range(11))

        if ylabel != 'RMSD':
            ax.set_ylim(0.0,1.0)
        else:
            ax.set_ylim(0.0,4.0)
            ax.set_xlim(0.0,10.0)
             
        for i, typ in enumerate(types_list):
            dat = dat_dict[typ]
            ax.plot(dat[xlabels[0]] , dat[ylabel],color=color[i] ,marker='o',label=typ)


        if ylabel == 'cov_10':
            ax.legend(bbox_to_anchor=(1.00,0.5))
        elif ylabel == 'cov_15':
            ax.legend(bbox_to_anchor=(1.00,0.5))
        else:
            ax.legend(bbox_to_anchor=(1.00,1.0))
        plt.savefig('%s_%s.png'%(method,ylabel))
        
