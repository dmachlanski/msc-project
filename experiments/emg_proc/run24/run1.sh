#!/bin/bash
# subjects: [4, 5, 6, 7, 8]
# sessions: [1, 2, 3, 4, 5]
# gestures: [finger, fist]

subjects1=(4 5 6 7)
subjects2=(8)
sessions1=(1 2 3 4 5)
sessions2=(1 2)
gestures=(finger fist)

for s in ${subjects1[*]}
do
    for sess in ${sessions1[*]}
    do
        for g in ${gestures[*]}
        do
            echo subject $s, session $sess, gesture $g
            python ../../../run_exp8.py -d /detop/temp/experiment2_force/processed/combined/ -s=$s --sess=$sess -g $g --n_delays=4 --Nmdl=350 -r=5 --Np=5 --km_h=40 --rk=60 --ds -v=0 -o /detop/temp/experiment2_force/decomposition/run1/
        done
    done
done

for s in ${subjects2[*]}
do
    for sess in ${sessions2[*]}
    do
        for g in ${gestures[*]}
        do
            echo subject $s, session $sess, gesture $g
            python ../../../run_exp8.py -d /detop/temp/experiment2_force/processed/combined/ -s=$s --sess=$sess -g $g --n_delays=4 --Nmdl=350 -r=5 --Np=5 --km_h=40 --rk=60 --ds -v=0 -o /detop/temp/experiment2_force/decomposition/run1/
        done
    done
done