subjects=(4 5 6 7 8)
gestures=(finger fist)
nums=(1 2 3 4 5 6)

for s in ${subjects[*]}
do
    for g in ${gestures[*]}
    do
        for n in ${nums[*]}
        do

            echo subject $s, gesture $g, number $n
            echo delays 0
            python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=$s --sess=1 -g $g -n=$n --n_delays=0 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 -v=0 --save_debug --peak_dist=40 -o ./run1/

            echo delays 8
            python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=$s --sess=1 -g $g -n=$n --n_delays=8 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 -v=0 --save_debug --peak_dist=40 -o ./run2/
        done
    done
done