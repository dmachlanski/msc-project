#!/bin/bash
# FILE: decompose_30mvc.sh
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N decompose_30mvc
#$ -pe smp 2

for subject in {2..15..3}
do
    echo "Subject $subject, 30% MVC"
    python3 decompose.py -d ../../data/sim/converted/ -s=$subject --mvc=30 --n_delays=16 -L=5 -v=0 -o ./run1/
done