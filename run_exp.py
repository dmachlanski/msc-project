"""
This script runs the entire decomposition pipeline, but only once. After discovering that there is a need to run decomposition more than once, this script has been abandoned. See 'run_exp_mul.py' for an updated version running decomposition multiple times and aggregating all the results together.
"""

import argparse
import numpy as np
from kmckc import KmCKC
from ckc import CKC
from hybrid import Hybrid
from time import time
from scipy.signal import find_peaks
from utils import test_similarity, get_cov_isi

def get_parser():
    parser = argparse.ArgumentParser()

    # Data-related args
    parser.add_argument('-d', type=str, dest='data_path')
    parser.add_argument('-s', type=int, dest='subject', default=4)
    parser.add_argument('--sess', type=int, dest='session', default=1)
    parser.add_argument('-g', type=str, dest='gesture', default='finger', choices=['finger', 'fist'])
    parser.add_argument('-n', type=int, dest='number', default=-1)
    parser.add_argument('--ds', dest='downsample', action='store_true', default=False)

    # Algorithm-related args
    parser.add_argument('-a', type=str, dest='alg', default='kmckc', choices=['ckc', 'kmckc', 'gckc', 'hybrid'])
    parser.add_argument('--n_delays', type=int, dest='n_delays', default=10)
    parser.add_argument('--delay_step', type=int, dest='delay_step', default=1)
    parser.add_argument('--Nmdl', type=int, dest='km_nmdl', default=10)
    parser.add_argument('-r', type=int, dest='km_r', default=10)
    parser.add_argument('--Np', type=int, dest='km_np', default=30)
    parser.add_argument('--km_h', type=int, dest='km_h', default=20)
    parser.add_argument('--ipt_th', type=int, dest='km_ipt_th', default=-1)
    parser.add_argument('--rk', type=int, dest='km_rk', default=200)
    parser.add_argument('--peak_dist', type=int, dest='peak_dist', default=1)
    parser.add_argument('--const_j', type=int, dest='const_j', default=30)
    parser.add_argument('--iter', type=int, dest='itermax', default=10)

    # Post-processing (getting spikes and grouping)
    parser.add_argument('--peak_thr_mul', type=float, dest='peak_thr_mul', default=3.0)
    parser.add_argument('--corr_thr', type=float, dest='corr_thr', default=0.1)
    parser.add_argument('--min_len', type=int, dest='min_len', default=10)

    # General
    parser.add_argument('-v', type=int, dest='verbose', default=0)
    parser.add_argument('--no_filter', dest='no_filter', action='store_true', default=False)
    parser.add_argument('--store_ipts', dest='store_ipts', action='store_true', default=False)
    parser.add_argument('--save_debug', dest='save_debug', action='store_true', default=False)
    parser.add_argument('-o', type=str, dest='out_path')

    return parser

def decompose(data, options):
    if options.alg == 'kmckc':
        alg = KmCKC(data, delays=options.n_delays, step=options.delay_step)
    elif options.alg == 'ckc':
        alg = CKC(data, delays=options.n_delays, step=options.delay_step)
    elif options.alg == 'hybrid':
        alg = Hybrid(data, delays=options.n_delays, step=options.delay_step)
    else:
        print('Other decomposition methods unavailable at the moment.')    

    start_t = time()
    ret_val = alg.decompose_alt(options)
    end_t = time()
    if options.verbose > 1: print(f"Decomposition time: {end_t - start_t}s")

    return ret_val
    
def ipts_to_spikes(ipts, x_std, dist):
    mus = []
    for ipt in ipts:
        mu, _ = find_peaks(ipt, height=np.std(ipt) * x_std, distance=dist)
        mus.append(mu)

    return mus

def keep_unique(data, n_samples, corr_th, min_len):
    unique = {}

    # Start with the longest train
    longest = data[0]
    longest_id = 0
    for id, d in enumerate(data):
        if len(longest) < len(d):
            longest = d
            longest_id = id
    unique[longest_id] = longest

    for i, d in enumerate(data):
        if len(d) < min_len:
            continue

        is_unique = True
        for k in unique:
            if test_similarity(d, unique[k], n_samples) >= corr_th:
                is_unique = False
                break

        if is_unique:
            unique[i] = d
        elif get_cov_isi(d) < get_cov_isi(unique[k]):
            unique.pop(k)
            unique[i] = d

    return list(unique.values()), list(unique.keys())

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    if options.verbose > 0: print("--- START ---")
    if options.verbose > 0: print(options)

    #channels = [10,11,12,13,14,15,16,19,20,21,22,23,24,25,28,29,30,31,32,33,34,37,38,39,40,41,42,43,46,47,48,49,50,51,52,55,56,57,58,59,60,61,64,65,66,67,68,69,70,73,74,75,76,77,78,79,82,83,84,85,86,87,88,91,92,93,94,95,96,97,100,101,102,103,104,105,106,109,110,111,112,113,114,115]

    # Load the data
    setting = f'emg_proc_s0{options.subject}_sess{options.session}_{options.gesture}'
    if options.number > 0: setting = setting + f'_{options.number}'
    data = np.load(f'{options.data_path}{setting}.npy')

    #data = data[channels]

    # Downsample from 4kHz to 2kHz (take every second sample)
    if options.downsample: data = data[:, ::2]

    # Decompose the signals into IPTs
    if options.verbose > 1: print('Decomposing...')
    if options.save_debug:
        ipts, cx, cxx_inv = decompose(data, options)
    else:
        ipts = decompose(data, options)
    if options.store_ipts: np.save(f'{options.out_path}{setting}_ipt', ipts)

    # Convert IPTs into spike trains
    if options.verbose > 1: print('Converting to spike trains...')
    mus = ipts_to_spikes(ipts, options.peak_thr_mul, options.peak_dist)
    np.save(f'{options.out_path}{setting}_mu', mus)

    if not options.no_filter:
        # Filter out duplicates
        if options.verbose > 1: print('Filtering out duplicates...')
        unique_mus, unique_ids = keep_unique(mus, len(ipts[0]), options.corr_thr, options.min_len)
        if options.verbose > 1: print(f'Kept {len(unique_mus)} out of {len(mus)}')
        np.save(f'{options.out_path}{setting}_unique', unique_mus)
        np.save(f'{options.out_path}{setting}_unique_ipt', [ipts[i] for i in unique_ids])

    if options.save_debug:
        np.save(f'{options.out_path}{setting}_cx', [cx[i] for i in unique_ids])
        np.save(f'{options.out_path}{setting}_cxx_inv', cxx_inv)
    
    if options.verbose > 0: print("--- END ---")