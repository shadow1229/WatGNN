#!/bin/sh

for tg in $(cat ./targets_cmp) ;
do
    echo $tg
    python waterRMSD.py ./ref/$tg.pdb ./WatGNN/$tg.pdb          > wkgb_log/$tg.dat
done

./summary_native.py  > ./native_summary.dat
