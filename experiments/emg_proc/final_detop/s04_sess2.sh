gestures=(finger fist)
nums=(1 2 3 4 5 6)

for g in ${gestures[*]}
do
    for n in ${nums[*]}
    do
        echo gesture $g, number $n
        python -W ignore ../../../run_exp_mul.py -d /detop/temp/experiment2_force/processed/ -s=4 --sess=2 -g $g -n=$n --store_duplicates -v=2 -o /detop/temp/experiment2_force/decomposition/final/
    done
done