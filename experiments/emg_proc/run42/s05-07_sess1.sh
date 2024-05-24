subjects=(5 6 7)
gestures=(finger fist)

for s in ${subjects[*]}
do
    for g in ${gestures[*]}
    do
        echo subject $s, gesture $g
        python ../../../run_exp.py -d /detop/temp/experiment2_force/processed/combined/ -s=$s --sess=1 -g $g --n_delays=0 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --min_len=300 -v=0 -o ./
    done
done