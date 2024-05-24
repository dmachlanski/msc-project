nums=(1 2 3)

for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=1 --delay_step=4 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --save_debug -v=0 -o ./run1/
done

for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=2 --delay_step=4 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --save_debug -v=0 -o ./run2/
done