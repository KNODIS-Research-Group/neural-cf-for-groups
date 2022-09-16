#!/bin/bash
OUTDIR='groups_2022_05_completeinfo'
#OUTDIR='groups_2022_04_good_index'
#OUTDIR='groups_2022_03_virtual_user'
OUTDIR='groups_2022_06_complete_varios_ds'

#DS='ml1m'
DS='ft'
#DS='anime'
#DS='netflix'


if [ "$DS" != "netflix" ]; then
    for m in $(ls -1v -d $OUTDIR/$DS/*.h5); do
        echo "EVAL Model: $m"
        python ncf-groups-eval.py --m $m
        #python ncf-groups-eval-virtual-user.py --m $m
        echo "Finished EVAL Model: $m"
        printf "\n\n\n\n"
    done
else
    for gs in $(seq 10 -1 2); do
        echo "Size by Size: $gs"
        for m in $(ls -1v -d $OUTDIR/$DS/*.h5); do
            echo "EVAL Model: $m"
            python ncf-groups-eval.py --m $m --groupsize $gs
            #python ncf-groups-eval-virtual-user.py --m $m
            echo "Finished EVAL Model: $m"
            printf "\n\n\n\n"
        done
    done
fi