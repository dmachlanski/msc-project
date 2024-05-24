::python -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/ -s=4 --sess=1 -g finger -n=1 --n_delays=8 -a kmckc_mod --Nmdl=150 -r=10 --Np=10 --km_h=10 --rk=60 --min_len=10 -L=1 -v=3 -o ./run24/

python -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/combined/ -s=7 --sess=1 -g finger -n=-1 --n_delays=0 --Nmdl=10 -r=5 --Np=5 --km_h=40 --rk=60 --min_len=10 --cov=0.3 -L=1 -v=4 -o ./run29/

::python -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/ -s=7 --sess=1 -g finger -n=1 --n_delays=0 -a hybrid --iter=10 --const_j=125 -r=10 --Np=10 --km_h=20 --min_len=10 --cov=0.3 -L=5 -v=2 -o ./run27/

::python -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/ -s=7 --sess=1 -g finger -n=1 --n_delays=0 -a ckc --iter=50 --const_j=125 --min_len=10 --cov=0.3 -L=5 -v=2 -o ./run27/

::python -W ignore ../../../run_exp_mul.py -d ../../../data/emg_proc/ -s=7 --sess=1 -g finger -n=1 --n_delays=0 -a fastICA --iter=10 --min_len=10 --cov=0.5 -L=1 -v=4 --no_whiten -o ./run28/