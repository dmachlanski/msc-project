import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.signal import find_peaks
import sys
sys.path.append("../")
from utils import test_similarity

def ipts_to_spikes(ipts, thr_mul, mode):
    mus = []
    for ipt in ipts:
        if mode == 'med':
            thr = thr_mul * np.median(np.abs(ipt) / 0.6745)
        else:
            thr = thr_mul * np.std(ipt)
        mu, _ = find_peaks(ipt, height=thr, distance=1)
        mus.append(mu)

    return mus

def keep_unique(data, n_samples, corr_th, min_len):
    unique = []
    ids = []

    for i, d in enumerate(data):
        if len(d) < min_len:
            continue

        if len(unique) < 1:
            unique.append(d)
            ids.append(i)
            continue

        is_unique = True
        for u in unique:
            sim = test_similarity(d, u, n_samples)
            if sim >= corr_th:
                is_unique = False
                break

        if is_unique:
            unique.append(d)
            ids.append(i)

    return unique, ids

with open('test_filter_thr.yaml') as f:
    params = yaml.load(f)

ipts = np.load(params['path'])
trains = ipts_to_spikes(ipts, params['peak_thr_mul'], params['peak_thr'])
n_samples = len(ipts[0])
unique, ids = keep_unique(trains, n_samples, params['threshold'], params['min_len'])

print(f'All: {len(ipts)}, kept: {len(unique)}')
print(ids)

plot_id = 1
if params['plot_trains']:
    f2 = plt.figure(plot_id)
    plot_id += 1
    plt.eventplot(unique, linelengths=0.8, linewidths=0.8)
    plt.tight_layout()
    f2.show()

if params['plot_heat']:
    f3 = plt.figure(plot_id)
    heatmap = np.empty((len(unique), len(unique)))
    for i, u1 in enumerate(unique):
        for j, u2 in enumerate(unique):
            if i == j:
                heatmap[i][j] = 0.0
            else:
                heatmap[i][j] = test_similarity(u1, u2, n_samples)
    print(f'Min: {np.min(heatmap)}, Max: {np.max(heatmap)}')
    plt.matshow(heatmap, fignum=f3.number)
    plt.xticks(range(heatmap.shape[0]), range(1, 1+heatmap.shape[0]), fontsize=14, rotation=45)
    plt.yticks(range(heatmap.shape[0]), range(1, 1+heatmap.shape[0]), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    f3.show()

input()