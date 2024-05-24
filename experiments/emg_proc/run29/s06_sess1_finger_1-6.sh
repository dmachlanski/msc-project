nums=(1 2 3 4 5 6)

for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp9.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=1 --Nmdl=350 -r=5 --Np=5 --km_h=40 --rk=60 --ds -v=0 -o ./
done