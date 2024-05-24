nums=(1 2 3 4 5 6)

for n in ${nums[*]}
do
    echo number $n
    echo delays 8
    python -W ignore ../../../run_exp_mul.py -d /detop/temp/experiment2_force/processed/ -s=7 --sess=1 -g finger -n=$n --n_delays=8 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --min_len=10 --cov=0.3 -L=5 -v=2 -o ./run1/

    echo delays 16
    python -W ignore ../../../run_exp_mul.py -d /detop/temp/experiment2_force/processed/ -s=7 --sess=1 -g finger -n=$n --n_delays=16 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --min_len=10 --cov=0.3 -L=5 -v=2 -o ./run2/

    echo delays 32
    python -W ignore ../../../run_exp_mul.py -d /detop/temp/experiment2_force/processed/ -s=7 --sess=1 -g finger -n=$n --n_delays=32 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --min_len=10 --cov=0.3 -L=5 -v=2 -o ./run3/
done