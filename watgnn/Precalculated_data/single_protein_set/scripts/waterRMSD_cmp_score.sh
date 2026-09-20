
for tg in $(cat ./targets_cmp) ;
do
    echo $tg
	
    python waterRMSD_score.py foldx wkgb_ref/$tg.pdb FoldX/$tg.pdb > result_score/FoldX/$tg.dat
    python waterRMSD_score.py rism wkgb_ref/$tg.pdb 3drism/$tg.pdb > result_score/3drism/$tg.dat
    python waterRMSD_score.py wkgb wkgb_ref/$tg.pdb wkgb/$tg.pdb > result_score/GalaxyWater-wKGB/$tg.dat
    python waterRMSD_score.py cnn wkgb_ref/$tg.pdb GalaxyWater-CNN/$tg.pdb > result_score/GalaxyWater-CNN/$tg.dat
    python waterRMSD_score.py gnn wkgb_ref/$tg.pdb watGNN/$tg.pdb > result_score/WatGNN/$tg.dat
done
./summary_native_score.py targets_cmp result_score/FoldX > summary_score/native_summary_FoldX.dat
./summary_native_score.py targets_cmp result_score/3drism > summary_score/native_summary_3drism.dat
./summary_native_score.py targets_cmp result_score/GalaxyWater-wKGB > summary_score/native_summary_GalaxyWater-wKGB.dat
./summary_native_score.py targets_cmp result_score/GalaxyWater-CNN > summary_score/native_summary_GalaxyWater-CNN.dat
./summary_native_score.py targets_cmp result_score/WatGNN > summary_score/native_summary_WatGNN.dat
python ./plot_wkgb_score.py
