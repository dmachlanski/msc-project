#!/bin/bash
# FILE: decompose_50mvc.sh
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N decompose_50mvc
#$ -pe smp 2

for subject in {3..15..3}
do
    echo "Subject $subject, 50% MVC"
    python3 decompose.py -d ../../data/sim/converted/ -s=$subject --mvc=50 --n_delays=16 -L=5 -v=0 -o ./run1/
done