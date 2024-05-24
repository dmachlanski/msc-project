nums=(1 2 3)

echo distance 10
for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=0 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --peak_dist=10 -v=0 -o ./run1/
done

echo distance 20
for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=0 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --peak_dist=20 -v=0 -o ./run2/
done

echo distance 40
for n in ${nums[*]}
do
    echo number $n
    python ../../../run_exp11.py -d /detop/temp/experiment2_force/processed/ -s=6 --sess=1 -g finger -n=$n --n_delays=0 --Nmdl=150 -r=10 --Np=10 --km_h=20 --rk=60 --peak_dist=40 -v=0 -o ./run3/
done