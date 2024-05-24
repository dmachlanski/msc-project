nums=(4 5 6)

for n in ${nums[*]}
do
    echo number $n
    python -W ignore ../../../run_exp_mul.py -d /detop/temp/experiment2_force/processed/ -s=7 --sess=1 -g finger -n=$n --n_delays=16 --Nmdl=350 -r=5 --Np=5 --km_h=40 --rk=60 --min_len=10 --cov=0.3 -L=5 -v=2 -o ./run2/
done