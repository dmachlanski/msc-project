#!/bin/bash
# FILE: test.sh
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N test
#$ -pe smp 2

python3 -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/ -s=4 --sess=1 -g finger -n=1 --Nmdl=10 -L=1 --n_delays=16 --store_duplicates -v=2 -o ./