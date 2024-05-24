nums=(1 2 3 4 5 6)

for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp3.py -d /detop/temp/experiment2_force/processed/ -s=4 --sess=1 -g finger -n=$n --n_delays=4 --delay_step=1 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 -v=0 -o ./
done