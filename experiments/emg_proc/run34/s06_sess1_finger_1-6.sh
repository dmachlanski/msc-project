nums=(1 2 3 4 5 6)

for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp9.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=8 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 -v=0 -o ./
done