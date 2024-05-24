#!/bin/bash
# FILE: s07_sess2.sh
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N s07_sess2
#$ -pe smp 2

gestures=(finger fist)
nums=(1 2 3 4 5 6)

for g in ${gestures[*]}
do
    for n in ${nums[*]}
    do
        echo gesture $g, number $n
        python3 -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/ -s=7 --sess=2 -g $g -n=$n --store_duplicates -v=2 -o ./results/
    done
done