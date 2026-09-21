import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib.font_manager import FontProperties #unicode
ANGSTROM = "A"
def read_dat(fpath):
    dat_t = []
    f = open(fpath,'r') 
    lines = f.readlines()
    for line in lines:
        lsp = line.split()
        if lsp[0].startswith('#'): 
            continue
        lsp_float = [float(lsp[i]) for i  in range(len(lsp))] 
        dat_t.append(lsp_float)
    dat_np_t = np.array(dat_t)
    dat_np = dat_np_t.transpose()
    return dat_np #dat_np[0]: prop [1]: acc, [2]: cov, [3]: rmsd

cuts = [0.5,1.0,1.5,2.0]
s_dl = 0
n_dl = 1

labels = ['Accuracy','Coverage','RMSD']
tts = ['test']
tts2 = {'train' :'$\mathrm{protein-compound}$ $\mathrm{comparison}$ $\mathrm{set}$ ($\mathrm{171}$ $\mathrm{structures}$)',
         'test' :'$\mathrm{protein-compound}$ $\mathrm{comparison}$ $\mathrm{set}$ ($\mathrm{171}$ $\mathrm{structures}$)'}


paths = ['WatGNN','GalaxyWater-CNN','3D-RISM'] 
dd    ={'WatGNN':'WatGNN', 
        'GalaxyWater-CNN':'GalaxyWater-CNN',
        '3D-RISM':'3D-RISM'
        }
color  = ['#000000',
          '#FF0000',
          '#00FF00',
          '#0000FF',
          '#000000',
          '#FF00FF',
          '#00FFFF',
          '#880088',
          '#008888',
          '#FF0000','#FF8800','#00FF00','#0000FF','#000000','#880088','#008888']
color2  = ['#FF0000','#00FF00','#0000FF','#000000','#880088','#008888']
plt.rc('mathtext', fontset='cm')

for tt in tts:
    for labelid in range(len(labels)):
        for cut in cuts:
            dat_dict = {}
            for p in paths:
                print(p)
                fpath = './summary/%s_test_c%3.1f.txt'%(p,cut)
                print (fpath)
                dat = read_dat(fpath)
                dat_dict[p] = dat 
            plt.rc('mathtext', fontset='cm')
            #plt.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
            #plt.rc('text',usetex=True)
            fig = plt.figure(figsize = (6,4),
                             facecolor = 'white',
                             edgecolor = 'black',
                             dpi  = 300
                            )
            prop = FontProperties(size=16) #unicode
            prop_title = FontProperties(size=14) #unicode
            #ax = fig.add_subplot(111)
            ax = fig.add_axes([0.14,0.15,0.80,0.74])
            ax.grid(visible=True, axis='both',linestyle='dotted',color='black')
            ax.legend(bbox_to_anchor=(1.00,1.0))
            title = '%s'%tts2[tt]
            ax.set_title(r'%s'%title,fontproperties=prop_title)
             
            if labelid == 0: #acc / cov
                ax.set_xlim(0.0,10.0)
                ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
                ax.set_ylim(0.0,1.0)
                ax.set_ylabel(r'$\mathrm{Mean}$ $\mathrm{precision ({%3.1f \AA})}$'%cut,fontproperties=prop)
                ax.set_xticks([1.0*i for i in range(11)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(dat[0] , 0.01*dat[1+labelid],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.8))

            elif labelid == 1: #acc / cov
                ax.set_xlim(0.0,10.0)
                ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
                ax.set_ylim(0.0,1.0)
                ax.set_ylabel(r'$\mathrm{Mean}$ $\mathrm{recall ({%3.1f \AA})}$'%cut,fontproperties=prop)
                ax.set_xticks([1.0*i for i in range(11)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(dat[0] , 0.01*dat[1+labelid],color=color[i] ,marker='o',label=dd[p])
                if cut == 0.5:
                    ax.legend(bbox_to_anchor=(1.00,0.95))
                else:
                    ax.legend(bbox_to_anchor=(1.00,0.4))
            elif labelid == 2: #RMSD 
                ax.set_xlim(0.0,10.0)
                ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
                ax.set_ylim(0.0,4.0)
                ax.set_ylabel(r'$\mathrm{RMSD ( {\AA } )}$',fontproperties=prop)
                ax.set_xticks([1.0*i for i in range(11)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(dat[0] , dat[1+labelid],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.8))
            else:
                ax.set_xlim(0.0,1.0)
                ax.set_xlabel(r'$\mathrm{Recall ({%3.1f \AA})}$'%cut,fontproperties=prop)
                ax.set_ylim(0.0,1.0)
                ax.set_ylabel(r'$\mathrm{Precision ({%3.1f \AA})}$'%cut,fontproperties=prop)
                ax.set_xticks([0.2*i for i in range(6)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(0.01*dat[2] , 0.01*dat[1],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.95))
                plt.gca().set_aspect("equal")

            if labelid != 3:
                plt.savefig('%s_%s_%3.1f.png'%(tt,labels[labelid][:3],cut))
            else:
                plt.savefig('%s_acccov_%3.1f.png'%(tt,cut))
        
