python -W ignore ../../../run_exp.py -d ../../../data/emg_proc/ -s=6 --sess=1 -g finger -n=1 --n_delays=0 --Nmdl=10 -r=10 --Np=10 --km_h=20 --rk=60 -v=2 -o ./run22/

::python ../../../run_exp.py -d ../../../data/emg_proc/ -s=6 --sess=1 -g finger -n=1 --n_delays=0 -a hybrid --iter=150 --const_j=200 -r=10 --Np=10 --km_h=20 --store_ipts --no_filter -v=2 -o ./run15/

::python ../../../run_exp.py -d ../../../data/emg_proc/ -s=6 --sess=1 -g finger -n=1 --n_delays=0 -a ckc --iter=300 --const_j=200 --store_ipts --no_filter -v=2 -o ./run17/