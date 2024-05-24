"""
This script performs the source tracking. It requires access to:
- Decomposition results (*_results.npy files).
- The EMG data to calculate the time lag from the spike train alignment in order to then select the best separation vector after its deconstruction.
- The ISCTEST method (which in turn requires matlab engine available locally).
"""

import argparse
import numpy as np
from tracking import isctest
from utils import get_source_lag

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', type=str, dest='data_path')
    parser.add_argument('-s', type=int, dest='subject')
    parser.add_argument('--sess', type=int, dest='session')
    parser.add_argument('-g', type=str, dest='gesture', choices=['finger', 'fist', 'all'])
    parser.add_argument('-K', type=int, dest='ext_factor', default=16)
    parser.add_argument('-a', type=float, dest='alpha', default=0.001)
    parser.add_argument('--align', dest='align', action='store_true', default=False)
    parser.add_argument('-r', type=str, dest='rec_path')
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('-v', type=int, dest='verbose', default=0)
    parser.add_argument('-o', type=str, dest='out_path')

    return parser

def get_info(options, gestures, repetitions):
    data = []
    max_components = n_samples = -1
    for gesture in gestures:
        for repetition in repetitions:
            path = '{}emg_proc_s0{}_sess{}_{}_{}_results.npy'.format(options.data_path, options.subject, options.session, gesture, repetition)
            try:
                results = np.load(path)
                if n_samples < 0:
                    n_samples = len(results[0]['sep'])
                n_components = len(results)
                if n_components > max_components:
                    max_components = n_components
            except:
                results = []
            data.append(results)
    return {'data': data, 'max_components': max_components, 'n_samples': n_samples}

def add_group(groups, t1, t2):
    added = False
    for group in groups:
        if t1 in group or t2 in group:
            group.add(t1)
            group.add(t2)
            added = True
            break
    if not added:
        groups.append({t1, t2})

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    if options.verbose > 0: print("--- START ---")
    if options.verbose > 0: print(options)

    gestures = [options.gesture] if options.gesture != 'all' else ['finger', 'fist']
    repetitions = range(1, 7)
    n_repetitions = len(repetitions)
    n_observations = options.ext_factor + 1

    info = get_info(options, gestures, repetitions)

    dim_0 = int(info['n_samples'] / n_observations)
    dim_1 = info['max_components'] if options.align else int(info['max_components'] * n_observations)
    dim_2 = len(info['data'])

    # Initialise the input matrix for ISCTEST
    isc_mat = np.zeros((dim_0, dim_1, dim_2))

    for i_c, contraction in enumerate(info['data']):
        for i_s, source in enumerate(contraction):
            v_mixed = source['sep'].flatten()
            ids = np.arange(0, len(v_mixed), n_observations)
            if options.align:
                gesture = gestures[i_c // n_repetitions]
                rep_id = (i_c % n_repetitions) + 1
                recording = np.load('{}emg_proc_s0{}_sess{}_{}_{}.npy'.format(options.rec_path, options.subject, options.session, gesture, rep_id))
                # Gets the lag from the alignment
                lag = get_source_lag(recording, source['peaks'], 75)
                # This can go out of expected bounds sometimes.
                # (which suggests the lag is not only due to the extension factor - interesting)
                lag_safe = min(max(-lag, 0), options.ext_factor)
                isc_mat[:, i_s, i_c] = v_mixed[ids + lag_safe]
            else:
                for lag in range(n_observations):
                    isc_mat[:, (i_s * n_observations) + lag, i_c] = v_mixed[ids + lag]

    isc_path = '{}isctest/'.format(options.root_path)
    clusters = isctest(isc_mat, isc_path, options.alpha)
    normalised_clusters = clusters if options.align else np.ceil(clusters / n_observations)

    # The output obtained from ISCTEST needs to be postprocessed.
    converted = []
    for cluster in normalised_clusters:
        new_cluster = []
        for i, element in enumerate(cluster):
            element_int = int(element)
            if element_int > 0:
                gesture = gestures[i // n_repetitions]
                rep_id = (i % n_repetitions) + 1
                item = (gesture, rep_id, element_int - 1)
                new_cluster.append(item)
        converted.append(new_cluster)
    
    grouped = []
    for cluster in converted:
        for i in range(len(cluster) - 1):
            add_group(grouped, cluster[i], cluster[i+1])

    # Probably the order is not important
    sorted_group = []
    for g in grouped:
        sorted_group.append(sorted(list(g)))

    if options.verbose > 1:
        print('Found {} group(s).'.format(len(sorted_group)))
        for sg in sorted_group:
            print(sg)

    np.save('{}emg_s0{}_sess{}_{}_common'.format(options.out_path, options.subject, options.session, options.gesture), sorted_group)

    if options.verbose > 0: print("--- END ---")