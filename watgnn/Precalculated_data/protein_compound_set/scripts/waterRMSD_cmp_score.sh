
for tg in $(cat ./pdbbind_exclude30.txt) ;
do
    echo $tg
	
    python waterRMSD_score.py rism ref/${tg}_water.pdb 3D-RISM/$tg.pdb > result_score/3drism/$tg.dat
    python waterRMSD_score.py cnn ref/${tg}_water.pdb GalaxyWater-CNN/$tg.pdb > result_score/GalaxyWater-CNN/$tg.dat
    python waterRMSD_score.py gnn ref/${tg}_water.pdb WatGNN/${tg}_wat_lig_all.pdb > result_score/WatGNN/$tg.dat

done
./summary_score.py pdbbind_exclude30.txt result_score/3drism > summary_score/native_summary_3drism.dat
./summary_score.py pdbbind_exclude30.txt result_score/GalaxyWater-CNN > summary_score/native_summary_GalaxyWater-CNN.dat
./summary_score.py pdbbind_exclude30.txt result_score/WatGNN > summary_score/native_summary_WatGNN.dat
python ./plot_pdbbind_score.py
