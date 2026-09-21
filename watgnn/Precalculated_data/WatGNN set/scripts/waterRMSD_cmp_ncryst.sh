#!/bin/sh

for tg in $(cat ./train.txt) ;
do
    echo $tg
    python waterRMSD_ncryst.py gnn ./gnn_newset_result_001_answer/$tg.pdb ./WatGNN_newset/$tg.pdb          > newset_log/$tg.dat
done

for tg in $(cat ./validation.txt) ;
do
    echo $tg
    python waterRMSD_ncryst.py gnn ./gnn_newset_result_001_answer/$tg.pdb ./WatGNN_newset/$tg.pdb          > newset_log/$tg.dat
done

for tg in $(cat ./test.txt) ;
do
    echo $tg
    python waterRMSD_ncryst.py gnn ./gnn_newset_result_001_answer/$tg.pdb ./WatGNN_newset/$tg.pdb          > newset_log/$tg.dat
done

./summary_native_score.py  train.txt      newset_log  > ./train_summary.dat
./summary_native_score.py  validation.txt newset_log  > ./validation_summary.dat
./summary_native_score.py  test.txt       newset_log  > ./test_summary.dat
