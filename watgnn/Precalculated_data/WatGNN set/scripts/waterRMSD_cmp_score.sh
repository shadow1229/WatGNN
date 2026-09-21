
for tg in $(cat ./validation.txt) ;
do
    echo $tg 
    python waterRMSD_score.py gnn ./gnn_newset_result_001_answer/$tg.pdb ./WatGNN_newset/$tg.pdb > result_score/WatGNN/$tg.dat
	
done
./summary_native_score.py validation.txt result_score/WatGNN > summary_score/native_summary_WatGNN.dat
python ./plot_newset_score.py