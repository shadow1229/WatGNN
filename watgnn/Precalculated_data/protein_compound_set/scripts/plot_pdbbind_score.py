import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib.font_manager import FontProperties #unicode
def read_dat(fpath):
    #wkgb/data(_cmp)/native(relaxed)_summary.dat
    #  N   Nprd scorecut |   RMSD    Ave    Med  pre0.5 pre1.0 pre1.5 rec0.5 rec1.0 rec1.5 f1_0.5 f1_1.0 f1_1.5
    f = open(fpath,'r')
    dump = []
    lines  = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        lsp = line.split()
        x = { 'n_cryst':float(lsp[0]),
              'n'    :float(lsp[1]),
              'scorecut':0.01*float(lsp[2]),
              'RMSD' :float(lsp[4]),
              'acc_05':float(lsp[7]),
              'acc_10':float(lsp[8]),
              'acc_15':float(lsp[9]),
              'cov_05':float(lsp[10]),
              'cov_10':float(lsp[11]),
              'cov_15':float(lsp[12]), 
              'f1_05':float(lsp[13]),
              'f1_10':float(lsp[14]),
              'f1_15':float(lsp[15])}
        dump.append(x)

    result_tr = []
    N_cryst = dump[0]['n_cryst'] 
    N_pred = dump[-1]['n']
    for item in dump:
        if item['n'] > N_pred:
            continue

        x = { 'n'    :float(item['n'])/ float(item['n_cryst']),
              'scorecut':item['scorecut'],
              'RMSD' :item['RMSD'],
              'acc_05':item['acc_05'],
              'acc_10':item['acc_10'],
              'acc_15':item['acc_15'], 
              'cov_05':item['cov_05'],
              'cov_10':item['cov_10'],
              'cov_15':item['cov_15'],
              'f1_05':item['f1_05'],
              'f1_10':item['f1_10'],
              'f1_15':item['f1_15'],}
        result_tr.append(x)

    result = {}
    for item in result_tr:
        for k in item.keys():
            if not k in result.keys():
                result[k] = []
            result[k].append(item[k])

    return result
    
labels = ['RMSD','acccov_05','acccov_10','acccov_15','f1_05','f1_10','f1_15']
labels_dict = { 'RMSD': ['n','RMSD'],
                'acccov_05' : ['cov_05','acc_05'],
                'acccov_10' : ['cov_10','acc_10'],
                'acccov_15' : ['cov_15','acc_15'],
                'f1_05' : ['scorecut','f1_05'],
                'f1_10' : ['scorecut','f1_10'],
                'f1_15' : ['scorecut','f1_15'] }   #'label':[xlabel,ylabel]    
cut_dict = {    'acccov_05' : 0.5,
                'acccov_10' : 1.0,
                'acccov_15' : 1.5,
                'f1_05' : 0.5,
                'f1_10' : 1.0,
                'f1_15' : 1.5,
            }                
types_dict = {'native' :'$\mathrm{protein-compound}$ $\mathrm{comparison}$ $\mathrm{set}$ '}


method_dict    ={'GalaxyWater-wKGB':'GalaxyWater-KGB','3drism':'3D-RISM','FoldX':'FoldX',
                 'GalaxyWater-CNN':'GalaxyWater-CNN','WatGNN':'WatGNN'}
method_list    = ['WatGNN','GalaxyWater-CNN','3drism']
method_list_f1    = ['WatGNN']
color  = ['#000000','#FF0000','#00FF00','#000000','#880088','#008888']
plt.rc('mathtext', fontset='cm')
#plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text',usetex=True)

for typ in types_dict.keys():
    dat_dict = {}
    for method in method_list:
        fpath = './summary_score/native_summary_%s.dat'%(method)
        print (fpath)
        dat = read_dat(fpath)
        dat_dict[method] = dat  #dat_dict: {'wkgb':{} ,}


        
    for label in labels:
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
        title = '%s $\mathrm{(171}$ $\mathrm{structures)}$'%types_dict[typ]
        ax.set_title(r'%s'%title,fontproperties=prop)

        if label in ['acccov_05','acccov_10','acccov_15']:
            cut = cut_dict[label] 
            ax.set_xlim(0.0,1.0)
            ax.set_xlabel(r'$\mathrm{Mean}$ $\mathrm{recall ({%3.1f \AA})}$'%cut,fontproperties=prop)
            ax.set_ylim(0.0,1.0)
            ax.set_ylabel(r'$\mathrm{Mean}$ $\mathrm{precision ({%3.1f \AA})}$'%cut,fontproperties=prop)
            ax.set_xticks([0.2*i for i in range(6)])
            ax.set_yticks([0.2*i for i in range(6)])
            for i, method in enumerate(method_list):
                dat = dat_dict[method]
                xlabel, ylabel = labels_dict[label]
                ax.plot(dat[xlabel] , dat[ylabel],color=color[i] ,marker='o',label=method_dict[method])
            ax.legend(bbox_to_anchor=(1.00,0.95),fontsize=8)
            plt.gca().set_aspect("equal")        
        
        elif label in ['f1_05','f1_10','f1_15']:
            cut = cut_dict[label] 
            ax.set_xlim(0.0,1.0)
            ax.set_xlabel(r'$\mathrm{Score}$ $\mathrm{cutoff}$',fontproperties=prop)
            ax.set_ylim(0.0,1.0)
            ax.set_ylabel(r'$\mathrm{Mean}$ $\mathrm{F1}$ $\mathrm{Score ({%3.1f \AA}}$ $\mathrm{criterion)}$'%cut,fontproperties=prop)
            for i, method in enumerate(method_list_f1):
                dat = dat_dict[method]
                xlabel, ylabel = labels_dict[label]
                ax.plot(dat[xlabel] , dat[ylabel],color=color[i] ,marker='o',label=method_dict[method])
            ax.legend(bbox_to_anchor=(1.00,0.95),fontsize=8)
            

        else:
            ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
            ax.set_ylabel(r'$\mathrm{RMSD}$ $\mathrm{( {\AA } )}$',fontproperties=prop)
            ax.set_ylim(0.0,4.0)
            ax.set_xlim(0.0,10.0) 
            for i, method in enumerate(method_list):
                dat = dat_dict[method]
                xlabel, ylabel = labels_dict[label]
                print("%s %s"%(dat[xlabel] , dat[ylabel]))
                ax.plot(dat[xlabel] , dat[ylabel],color=color[i] ,marker='o',label=method_dict[method])

        plt.savefig('%s.png'%(label))
        