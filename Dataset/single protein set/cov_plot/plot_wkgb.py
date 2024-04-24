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
types_dict = {'native' :'$\mathrm{single}$ $\mathrm{protein}$ $\mathrm{comparison}$ $\mathrm{set}$'}
#        'relaxed':'$\mathrm{perturbed}$ $\mathrm{structure}$ $\mathrm{set}$'}

method_dict    ={'wkgb':'GalaxyWater-KGB','3drism':'3D-RISM','foldX':'FoldX',
                 'GWCNN':'GalaxyWater-CNN','GWGNN':'WatGNN'}
method_list    = ['GWGNN','GWCNN','wkgb','3drism','foldX']

color  = ['#000000','#FF0000','#FF8800','#00FF00','#0000FF','#000000','#880088','#008888']
plt.rc('mathtext', fontset='cm')
#plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text',usetex=True)

for typ in types_dict.keys():
    dat_dict = {}
    for method in method_list:
        fpath = './%s_wkgb.dat'%(method)
        print (fpath)
        dat = read_dat(fpath)
        dat_dict[method] = dat  #dat_dict: {'wkgb':{} ,}
        
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
        title = '%s $\mathrm{(92}$ $\mathrm{structures)}$'%types_dict[typ]
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
             
        for i, method in enumerate(method_list):
            dat = dat_dict[method]
            ax.plot(dat[xlabels[0]] , dat[ylabel],color=color[i] ,marker='o',label=method_dict[method])


        if ylabel == 'cov_10':
            ax.legend(bbox_to_anchor=(1.00,0.33),fontsize=8)
        elif ylabel == 'cov_15':
            ax.legend(bbox_to_anchor=(1.00,0.33),fontsize=8)
        else:
            ax.legend(bbox_to_anchor=(1.00,1.0),fontsize=8)
        plt.savefig('%s.png'%(ylabel))
        
