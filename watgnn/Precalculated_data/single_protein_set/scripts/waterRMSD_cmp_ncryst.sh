
for tg in $(cat ./targets_cmp) ;
do
    echo $tg

    python waterRMSD_ncryst.py wkgb_ref/$tg.pdb 3drism/$tg.pdb > result/3drism/$tg.dat
    python waterRMSD_ncryst.py wkgb_ref/$tg.pdb FoldX/$tg.pdb > result/FoldX/$tg.dat
    python waterRMSD_ncryst.py wkgb_ref/$tg.pdb GalaxyWater-CNN/$tg.pdb > result/GalaxyWater-CNN/$tg.dat
    python waterRMSD_ncryst.py wkgb_ref/$tg.pdb wkgb/$tg.pdb > result/GalaxyWater-wKGB/$tg.dat
    python waterRMSD_ncryst.py wkgb_ref/$tg.pdb watGNN/$tg.pdb > result/WatGNN/$tg.dat
done

./summary_native.py targets_cmp result/3drism > summary/native_summary_3drism.dat
./summary_native.py targets_cmp result/FoldX > summary/native_summary_FoldX.dat
./summary_native.py targets_cmp result/GalaxyWater-CNN > summary/native_summary_GalaxyWater-CNN.dat
./summary_native.py targets_cmp result/GalaxyWater-wKGB > summary/native_summary_GalaxyWater-wKGB.dat
./summary_native.py targets_cmp result/WatGNN > summary/native_summary_WatGNN.dat

