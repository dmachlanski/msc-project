#!/bin/bash
# FILE: decompose_10mvc.sh
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N decompose_10mvc
#$ -pe smp 2

for subject in {1..15..3}
do
    echo "Subject $subject, 10% MVC"
    python3 decompose.py -d ../../data/sim/converted/ -s=$subject --mvc=10 --n_delays=16 -L=5 -v=0 -o ./run1/
done