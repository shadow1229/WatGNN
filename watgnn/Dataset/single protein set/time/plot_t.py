import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib.font_manager import FontProperties #unicode
def read_dat(fpath):
    #wkgb/data(_cmp)/native(relaxed)_summary.dat
    #  N |   RMSD    Ave    Med   f<0.5  f<1.0
    f = open(fpath,'r')
    x = []
    y = []
    lines  = f.readlines()
    for line in lines:
        lsp  = line.split()
        res  = int(lsp[0])
        logt = math.log10(float(lsp[1]))
        x.append(res)
        y.append(logt)
    return x,y
for i in range(1):
    for j in range(1): #16_6
        xlabels = ['n']
        ylabels = ['RMSD','XXXX','coverage']
        tts  = ['native','relaxed']
        tts2 = {'native' :'$\mathrm{crystal}$ $\mathrm{structure}$ $\mathrm{set}$',
                'relaxed':'$\mathrm{perturbed}$ $\mathrm{structure}$ $\mathrm{set}$'}
        #tts = ['native']
        ANGSTROM = "A"
        #dir_dict = {'data_cmp':['wkgb', '3drism' , 'foldX','final_cnn2'],
        #            'data':['wkgb','wkgb.woAll' , 'wkgb.woReff', 'wkgb.woVirt','final_cnn2']}
        color2  = ['#FF0000','#00FF00','#0000FF','#000000']
        rism_x,rism_y = read_dat('rism_rest.dat')
        fold_x,fold_y = read_dat('foldx_rest.dat')
        wkgb_x,wkgb_y = read_dat('wkgb_rest.dat')
        cnn_x,cnn_y = read_dat('cnn_rest.dat')
        
        gnn_x, gnn_y = read_dat('gnn_rest.dat')
        cnn_cpu_x,cnn_cpu_y = read_dat('cnn_cpu_rest.dat')
        
        plt.rc('mathtext', fontset='cm')
        #plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
        #plt.rc('text',usetex=True)
        fig = plt.figure(figsize = (6,4),
                         facecolor = 'white',
                         edgecolor = 'black',
                         dpi  = 300
                        )
        prop = FontProperties(size=16) #unicode
        prop2 = FontProperties(size=10) #unicode
        #ax = fig.add_subplot(111)
        ax = fig.add_axes([0.14,0.15,0.80,0.74])
        ax.grid(visible=True, axis='both',linestyle='dotted',color='black') #b-> visible
        title = '%s $\mathrm{(92}$ $\mathrm{targets)}$'%tts2['native']
        ax.set_title(r'%s'%title,fontproperties=prop)
        
        ax.set_xlim(0.0,500.0)
        ax.set_ylim(-2.0,4.0)
        ax.set_xlabel(r'$\mathrm{Protein}$ $\mathrm{size}$ $\mathrm{(number}$ $\mathrm{of}$ $\mathrm{amino}$ $\mathrm{acids)}$',fontproperties=prop)
        ax.set_ylabel(r'$\mathrm{log}_{10}\mathrm{(time)}$ $\mathrm{(s)}$',fontproperties=prop)

        ax.scatter(gnn_x , gnn_y,color='#000000' ,marker='o',label='WatGNN')         
        ax.scatter(cnn_x , cnn_y,color='#FF0000' ,marker='o',label='GWCNN')
        #ax.scatter(cnn_cpu_x , cnn_cpu_y,color='#000000' ,marker='o',label='GW2_CPU') 
        #ax.scatter(wkgb_x , wkgb_y,color='#FF8800' ,marker='o',label='GW_wKGB')
        ax.scatter(rism_x , rism_y,color='#00FF00' ,marker='o',label='3D-RISM')
        ax.scatter(fold_x , fold_y,color='#0000FF' ,marker='o',label='FoldX')
        ax.legend(bbox_to_anchor=(0.95,0.32),fontsize=8)
        plt.savefig('time.png')

