#!/bin/sh

for tg in $(cat ./targets_cmp) ;
do
    echo $tg
    python waterRMSD.py ../wkgb_ref/$tg.pdb ../gnn_result_wkgb_patch_sorted/$tg.pdb          > wkgb_log/$tg.dat
done

./summary_native.py  > ./native_summary.dat
