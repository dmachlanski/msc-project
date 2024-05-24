import argparse
import numpy as np
import pandas as pd
from kmckc import KmCKC
from time import time
from utils import merge_spikes, get_conf_mat
import elephant.conversion as conv

def get_parser():
    parser = argparse.ArgumentParser()

    # Data-related args
    parser.add_argument('-d', type=str, dest='data_path', default=None)
    parser.add_argument('--n_cells', type=int, dest='n_cells', default=5)
    parser.add_argument('--electrode', type=str, dest='electrode', default='Neuronexus-32', choices=['Neuronexus-32', 'Neuropixels-128'])
    parser.add_argument('--t_sec', type=int, dest='t_sec', default=40)
    parser.add_argument('--noise', type=int, dest='noise', default=0)
    parser.add_argument('--freq', type=int, dest='freq', default=20000)

    # Algorithm-related args
    parser.add_argument('-a', type=str, dest='alg', default='kmckc', choices=['ckc', 'kmckc', 'gckc'])
    parser.add_argument('--n_delays', type=int, dest='n_delays', default=10)
    parser.add_argument('--delay_step', type=int, dest='delay_step', default=1)
    parser.add_argument('--Nmdl', type=int, dest='km_nmdl', default=10)
    parser.add_argument('-r', type=int, dest='km_r', default=10)
    parser.add_argument('--Np', type=int, dest='km_np', default=30)
    parser.add_argument('--km_h', type=int, dest='km_h', default=20)
    parser.add_argument('--ipt_th', type=int, dest='km_ipt_th', default=200)
    parser.add_argument('--rk', type=int, dest='km_rk', default=200)

    # General
    parser.add_argument('--merge_window', type=int, dest='merge_window', default=10)
    parser.add_argument('--eval_offset', type=int, dest='eval_offset', default=2)
    parser.add_argument('-v', type=int, dest='verbose', default=0)

    return parser

def decompose(data, options):
    if options.alg == 'kmckc':
        alg = KmCKC(data, delays=options.n_delays, step=options.delay_step)
    else:
        print('Other decomposition methods unavailable at the moment.')    

    start_t = time()
    mus = alg.decompose_alt(options)
    end_t = time()
    if options.verbose > 0: print(f"Decomposition time: {end_t - start_t}s")

    return merge_spikes(mus, options.merge_window, x.shape[1] + options.n_delays)
    

def evaluate(mus, trains, options):
    results = {'pred_id': [], 'true_id': [], 'recall': [], 'precision': [], 'tp': [], 'fn': [], 'fp': []}
    for j, mu in enumerate(mus):
        if len(mu) < 1:
            continue
        for i, t in enumerate(trains):
            tp, fn, fp = get_conf_mat(t, mu, options.eval_offset, x.shape[1])
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            results['pred_id'].append(j)
            results['true_id'].append(i)
            results['recall'].append(recall)
            results['precision'].append(precision)
            results['tp'].append(tp)
            results['fn'].append(fn)
            results['fp'].append(fp)

    return results

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    if options.verbose > 0: print("--- START ---")

    # Load the data
    setting = f'rec_{options.n_cells}cells_{options.electrode}_{options.t_sec}sec_{options.noise}uV'
    x = np.load(f'{options.data_path}/{setting}/recordings.npy')

    # Decompose and save reconstructed spike trains
    mus = decompose(x, options)
    np.save(f'{setting}_results', mus)
    
    # Get the ground truth and convert it to the expected format
    trains_raw = np.load(f'{options.data_path}/{setting}/spiketrains.npy', allow_pickle=True)
    trains = []

    bins = options.freq * options.t_sec
    for t in trains_raw:
        bst = conv.BinnedSpikeTrain(t, num_bins=bins)
        trains.append(bst.spike_indices[0])

    # Evaluate obtained MUs against the gold standard and save the results to a file
    eval_results = evaluate(mus, trains, options)
    df = pd.DataFrame(eval_results)
    df.to_csv(f'{setting}_eval.csv', index=False)
    
    if options.verbose > 0: print("--- END ---")