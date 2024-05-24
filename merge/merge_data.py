"""
This script was used to combine individual trials of a given session into one long recording (but including only one movement type).
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()

# Data-related args
parser.add_argument('-d', type=str, dest='data_path')
parser.add_argument('-s', type=int, dest='subject', default=4)
parser.add_argument('--sess', type=int, dest='session', default=1)
parser.add_argument('-g', type=str, dest='gesture', default='finger', choices=['finger', 'fist'])
parser.add_argument('-n', type=int, dest='number', nargs='+')
parser.add_argument('-o', type=str, dest='out_path')

options = parser.parse_args()

setting = f'emg_proc_s0{options.subject}_sess{options.session}_{options.gesture}'

data_combined = None
for i, n in enumerate(options.number):
    x = np.load(f'{options.data_path}{setting}_{n}.npy')
    data_combined = np.hstack((data_combined, x)) if i > 0 else x

np.save(f'{options.out_path}{setting}', data_combined)