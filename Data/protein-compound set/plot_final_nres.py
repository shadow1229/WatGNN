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

labels = ['Accuracy','Coverage','RMSD','Acc_Cov']
#tts = ['train','test']
tts = ['test']
tts2 = {'train' :'$\mathrm{Protein-compound}$ $\mathrm{complex}$ $\mathrm{training}$ $\mathrm{set}$ ($\mathrm{370}$ $\mathrm{structures}$)',
         'test' :'$\mathrm{Protein-compound}$ $\mathrm{complex}$ $\mathrm{test}$ $\mathrm{set}$ ($\mathrm{370}$ $\mathrm{structures}$)'}

#paths = ['lig_lig_all','lig_nofg_lig_all','new_cnn11_nofg_all']
#
#dd    ={'lig_lig_all':'CNN_lig', 
#        'lig_nofg_lig_all':'CNN_lig.w/o SYBYL',
#        'new_cnn11_nofg_all':'CNN.w/o SYBYL',
#        }
paths = ['lig_lig_50_all','lig_nofg_lig_50_all','new_cnn11_nofg_50_all', 
         'lig_nofg_fullgrid_lig_50_all','mc_refine_20min_lig_50_all',
         'mc_refine_15_lig_50_all','mc_refine_20_lig_50_all',
         'grad_refine_15_lig_50_all','grad_refine_20_lig_50_all',
         
         ] #,'lig_nofg_fullgrid_2_lig_50_all' ]
paths = ['gnn_lig_50_all','cnn_lig_50_all','3drism_50_all'] #,'lig_nofg_fullgrid_2_lig_50_all' ]
#fullgrid_2: merge grid + change water placement method
#original: remove score(to 0) in protein voxel -> convolution with kernel
#fullgrid_2: convolution with kernel -> remove score(to 0) in protein voxel
#result: fullgrid and fullgrid_2 showed very similar result
dd    ={'gnn_lig_50_all':'WatGNN', 
        'cnn_lig_50_all':'GalaxyWater-CNN',
        '3drism_50_all':'3D-RISM'
        }
color  = ['#000000',
          '#FF0000',
          #'#FF8800',
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
                fpath = './%s_c%3.1f_dlv_%02d.txt'%(p,cut,0)
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
                ax.set_ylabel(r'$\mathrm{Accuracy ({%3.1f \AA})}$'%cut,fontproperties=prop)
                #ax.set_ylabel(r'$\mathrm{Coverage ({%3.1 A})}$'%cut,fontproperties=prop)
                ax.set_xticks([1.0*i for i in range(11)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(dat[0] , 0.01*dat[1+labelid],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.4))
            elif labelid == 1: #acc / cov
                ax.set_xlim(0.0,10.0)
                ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
                ax.set_ylim(0.0,1.0)
                if cut == 1.0:
                    ax.set_ylabel(r'$\mathrm{Coverage}$',fontproperties=prop)
                else:
                    ax.set_ylabel(r'$\mathrm{Coverage ({%3.1f \AA})}$'%cut,fontproperties=prop)
                #ax.set_ylabel(r'$\mathrm{Coverage ({%3.1 A})}$'%cut,fontproperties=prop)
                ax.set_xticks([1.0*i for i in range(11)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(dat[0] , 0.01*dat[1+labelid],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.4))
            elif labelid == 2: #RMSD 
                ax.set_xlim(0.0,10.0)
                ax.set_xlabel(r'$N_{\mathrm{pred}}/N_{\mathrm{cryst}}$',fontproperties=prop)
                ax.set_ylim(0.0,4.0)
                ax.set_ylabel(r'$\mathrm{RMSD ( {\AA } )}$',fontproperties=prop)
                #ax.set_ylabel(r'$\mathrm{RMSD ( {A } )}$',fontproperties=prop)
                ax.set_xticks([1.0*i for i in range(11)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(dat[0] , dat[1+labelid],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.4))
            else:
                ax.set_xlim(0.0,1.0)
                ax.set_xlabel(r'$\mathrm{Coverage ({%3.1f \AA})}$'%cut,fontproperties=prop)
                #ax.set_xlabel(r'$\mathrm{Coverage ({%3.1 A})}$'%cut,fontproperties=prop)
                ax.set_ylim(0.0,1.0)
                ax.set_ylabel(r'$\mathrm{Accuracy ({%3.1f \AA})}$'%cut,fontproperties=prop)
                #ax.set_ylabel(r'$\mathrm{Accuracy ({%3.1 A})}$'%cut,fontproperties=prop)
                ax.set_xticks([0.5*i for i in range(7)])
                for i, p in enumerate(paths):
                    dat = dat_dict[p]
                    ax.plot(0.01*dat[2] , 0.01*dat[1],color=color[i] ,marker='o',label=dd[p])
                ax.legend(bbox_to_anchor=(1.00,0.4))

            if labelid != 3:
                plt.savefig('%s_%s_%3.1f.png'%(tt,labels[labelid][:3],cut))
            else:
                plt.savefig('%s_acccov_%3.1f.png'%(tt,cut))
        
