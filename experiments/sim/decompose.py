import argparse
import numpy as np
import sys
sys.path.append("../../")
from kmckc import KmCKC
from time import time
from utils import get_peaks

def get_parser():
    parser = argparse.ArgumentParser()

    # Data-related args
    parser.add_argument('-d', type=str, dest='data_path')
    parser.add_argument('-s', type=int, dest='subject', default=1)
    parser.add_argument('--mvc', type=int, dest='mvc')
    parser.add_argument('--ds', dest='downsample', action='store_true', default=False)

    # Algorithm-related args
    parser.add_argument('--n_delays', type=int, dest='n_delays', default=16)
    parser.add_argument('--delay_step', type=int, dest='delay_step', default=1)
    parser.add_argument('--Nmdl', type=int, dest='km_nmdl', default=350)
    parser.add_argument('-r', type=int, dest='km_r', default=5)
    parser.add_argument('--Np', type=int, dest='km_np', default=5)
    parser.add_argument('--km_h', type=int, dest='km_h', default=40)
    parser.add_argument('--rk', type=int, dest='km_rk', default=60)

    # General
    parser.add_argument('-v', type=int, dest='verbose', default=0)
    parser.add_argument('-L', type=int, dest='loops', default=5)
    parser.add_argument('-o', type=str, dest='out_path')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    if options.verbose > 0: print("--- START ---")
    if options.verbose > 0: print(options)

    # Load the data
    setting = f'S{options.subject}_{options.mvc}MVC'
    data = np.load(f'{options.data_path}{setting}_rec.npy')

    # Downsample from 4kHz to 2kHz (take every second sample)
    if options.downsample: data = data[:, ::2]

    # Generates delayed repetitions
    alg = KmCKC(data, delays=options.n_delays, step=options.delay_step, whiten=True)

    times = []
    trains = []
    for _ in range(options.loops):
        start_t = time()
        # Decompose the signals into IPTs
        results = alg.decompose_alt(options)
        end_t = time()

        # IPTs -> peaks
        peaks = get_peaks(results, 'fixed_sq')

        times.append(end_t - start_t)
        trains.append(peaks)

    np.save(f'{options.out_path}{setting}_trains.npy', trains)
    np.save(f'{options.out_path}{setting}_times.npy', times)
    
    if options.verbose > 0: print("--- END ---")