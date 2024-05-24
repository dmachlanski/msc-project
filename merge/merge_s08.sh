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
            python merge_data.py -d /detop/temp/experiment2_force/processed/ -s=$s --sess=$sess -g $g -n 1 2 3 4 5 6 -o /detop/temp/experiment2_force/processed/combined/
        done
    done
done