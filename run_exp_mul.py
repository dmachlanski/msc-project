"""
This is the main script for running the entire decomposition pipeline. It performs all the necessary steps, that is, preprocessing, the decomposition itself, and the postprocessing (IPT conversion, quality filtering and duplicate filtering). Spike train alignment was added later in the project and is not part of this script.
"""

import argparse
import numpy as np
from kmckc import KmCKC
from kmckc_mod import KmCKC_mod
from ckc import CKC
from hybrid import Hybrid
from utils import quality_filter_alt, duplicate_filter

BAD_CHANNELS = {4: [54],
                5: [49],
                6: [44, 53, 60, 62, 71, 80, 89, 95, 98],
                7: [116],
                8: [95, 116]}

def get_parser():
    parser = argparse.ArgumentParser()

    # Data-related args
    parser.add_argument('-d', type=str, dest='data_path')
    parser.add_argument('-s', type=int, dest='subject', default=4)
    parser.add_argument('--sess', type=int, dest='session', default=1)
    parser.add_argument('-g', type=str, dest='gesture', default='finger', choices=['finger', 'fist'])
    parser.add_argument('-n', type=int, dest='number', default=-1)
    parser.add_argument('--ds', dest='downsample', action='store_true', default=False)
    parser.add_argument('--noref', dest='noref', action='store_true', default=False)

    # Algorithm-related args
    parser.add_argument('-a', type=str, dest='alg', default='kmckc', choices=['ckc', 'kmckc', 'kmckc_mod', 'gckc', 'hybrid', 'fastICA'])
    parser.add_argument('--n_delays', type=int, dest='n_delays', default=16)
    parser.add_argument('--delay_step', type=int, dest='delay_step', default=1)
    parser.add_argument('--no_whiten', dest='no_whiten', action='store_true', default=False)
    parser.add_argument('--Nmdl', type=int, dest='km_nmdl', default=350)
    parser.add_argument('-r', type=int, dest='km_r', default=5)
    parser.add_argument('--Np', type=int, dest='km_np', default=5)
    parser.add_argument('--km_h', type=int, dest='km_h', default=40)
    parser.add_argument('--ipt_th', type=int, dest='km_ipt_th', default=-1)
    parser.add_argument('--rk', type=int, dest='km_rk', default=60)
    parser.add_argument('--peak_dist', type=int, dest='peak_dist', default=1)
    parser.add_argument('--const_j', type=int, dest='const_j', default=30)
    parser.add_argument('--iter', type=int, dest='itermax', default=10)

    # Post-processing (getting spikes and grouping)
    parser.add_argument('--peak_thr_mul', type=float, dest='peak_thr_mul', default=3.0)
    # 'train': 0.1, 'vector': 0.8
    parser.add_argument('--corr_thr', type=float, dest='corr_thr', default=0.1)
    parser.add_argument('--min_len', type=int, dest='min_len', default=10)
    parser.add_argument('--sil', type=float, dest='sil', default=0.9)
    parser.add_argument('--cov', type=float, dest='cov', default=0.3)
    parser.add_argument('--group_mode', type=str, dest='group_mode', default='train', choices=['vector', 'train'])

    # General
    parser.add_argument('-v', type=int, dest='verbose', default=0)
    parser.add_argument('-L', type=int, dest='loops', default=5)
    parser.add_argument('--no_filter', dest='no_filter', action='store_true', default=False)
    parser.add_argument('--store_duplicates', dest='store_duplicates', action='store_true', default=False)
    parser.add_argument('-o', type=str, dest='out_path')

    return parser

def get_algorithm(data, options):
    if options.alg == 'kmckc':
        alg = KmCKC(data, delays=options.n_delays, step=options.delay_step, whiten=not options.no_whiten)
    elif options.alg == 'kmckc_mod':
        alg = KmCKC_mod(data, delays=options.n_delays, step=options.delay_step, whiten=not options.no_whiten)
    elif options.alg == 'ckc':
        alg = CKC(data, delays=options.n_delays, step=options.delay_step, whiten=not options.no_whiten)
    elif options.alg == 'hybrid':
        alg = Hybrid(data, delays=options.n_delays, step=options.delay_step, whiten=not options.no_whiten)
    else:
        print('Other decomposition methods unavailable at the moment.')

    return alg

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    if options.verbose > 0: print("--- START ---")
    if options.verbose > 0: print(options)

    # Load the data
    setting = 'emg_'
    if options.noref: setting += f'noREF_'
    setting += f'proc_s0{options.subject}_sess{options.session}_{options.gesture}'
    if options.number > 0: setting += f'_{options.number}'
    data = np.load(f'{options.data_path}{setting}.npy')

    # Delete noisy channels
    data = np.delete(data, BAD_CHANNELS[options.subject], 0)

    # Downsample from 4kHz to 2kHz (take every second sample)
    if options.downsample:
        data = data[:, ::2]
        fs = 2048
    else:
        fs = 4096

    all_results = []
    method = get_algorithm(data, options)

    for _ in range(options.loops):
        # Decompose the signals into IPTs
        results = method.decompose_alt(options)

        # Keep only quality sources
        quality_res = quality_filter_alt(results, options.cov, options.min_len, 'fixed_sq', options.peak_thr_mul, fs)
        if options.verbose > 1: print(f'Keeping {len(quality_res)} quality sources')

        all_results = all_results + quality_res

    if options.store_duplicates: np.save(f'{options.out_path}{setting}_results_all', all_results)

    # Filter out duplicates
    best_res = duplicate_filter(all_results, len(results[0]['ipt']), options.corr_thr, options.group_mode) if len(all_results) > 0 else []

    if options.verbose > 1: print(f'Keeping {len(best_res)} unique sources (out of {len(all_results)})')
    if len(best_res) > 0:
        np.save(f'{options.out_path}{setting}_results', best_res)
        for source in best_res:
            print(f"CoV={source['cov']:.3f}, Peaks={len(source['peaks'])}")
    else:
        print('Found 0 sources. Nothing to save.')
    
    if options.verbose > 0: print("--- END ---")