#!/bin/bash
# subjects: [4, 5, 6, 7, 8]
# sessions: [1, 2, 3, 4, 5, 6]
# gestures: [finger, fist]

subjects=(8)
sessions=(1 2)
gestures=(finger fist)

for s in ${subjects[*]}
do
    for sess in ${sessions[*]}
    do
        for g in ${gestures[*]}
        do
            echo subject $s, session $sess, gesture $g
            python ../../../run_exp7.py -d /detop/temp/experiment2_force/processed/combined/ -s=$s --sess=$sess -g $g --n_delays=4 --delay_step=1 --Nmdl=350 -r=5 --Np=5 --km_h=40 --rk=60 -v=0 -o /detop/temp/experiment2_force/decomposition/run1/
        done
    done
done