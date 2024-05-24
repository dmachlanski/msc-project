#!/bin/bash
#cells: [5, 10, 20, 30, 40, 50]
#electrodes: [Neuronexus-32, Neuropixels-128]
#time: [40, 60, 90, 120]
#noise: [0, 10, 20, 30]
#rec_{cells}cells_{electrodes}_{time}sec_{noise}uV

cells=(5 10 20 30 40 50)
electrodes=(Neuronexus-32 Neuropixels-128)
time=(40 60 90 120)
noise=(0 10 20 30)

for c in ${cells[*]}
do
    for e in ${electrodes[*]}
    do
        for t in ${time[*]}
        do
            for n in ${noise[*]}
            do
                echo "--n_cells=$c --electrode $e --t_sec=$t --noise=$n"
            done
        done
    done
done