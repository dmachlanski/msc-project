nums=(1 2 3 4 5 6)

for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp10.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=0 --Nmdl=150 -r=10 --Np=30 --km_h=20 --rk=200 -v=0 -o ./
done