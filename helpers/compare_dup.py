import numpy as np
import matplotlib.pyplot as plt
import yaml
from time import time
import sys
sys.path.append("../")
from utils import spike_similarity, test_similarity

def keep_unique_old(mus, sample_size, l_sim, s_sim, neighbours):
    unique = []
    ids = []

    for i, mu in enumerate(mus):
        if len(mu) < 1:
            continue

        if len(unique) < 1:
            unique.append(mu)
            ids.append(i)
            continue
    
        is_unique = True
        mu_length = len(mu)
        for d in unique:
            d_length = len(d)
            length_similarity = min(mu_length, d_length)/max(mu_length, d_length)

            if length_similarity < l_sim:
                continue
            elif spike_similarity(mu, d, neighbours, sample_size) >= s_sim:
                is_unique = False
                break

        if is_unique:
            unique.append(mu)
            ids.append(i)

    return unique, ids

def keep_unique_new(data, n_samples, corr_th, n):
    unique = []
    ids = []

    for i, d in enumerate(data):
        if len(d) < 1:
            continue

        if len(unique) < 1:
            unique.append(d)
            ids.append(i)
            continue

        is_unique = True
        for u in unique:
            if test_similarity(d, u, n_samples, 'simple', n) >= corr_th or test_similarity(u, d, n_samples, 'simple', n) >= corr_th:
                is_unique = False
                break

        if is_unique:
            unique.append(d)
            ids.append(i)

    return unique, ids

def keep_unique_weighted(data, n_samples, corr_th, min_len):
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
            if test_similarity(d, u, n_samples) >= corr_th:
                is_unique = False
                break

        if is_unique:
            unique.append(d)
            ids.append(i)

    return unique, ids

def keep_unique_max(data, n_samples, corr_th, min_len):
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
        elif len(d) > len(unique[k]):
            unique.pop(k)
            unique[i] = d

    return unique.values(), unique.keys()

with open('compare_dup.yaml') as f:
    params = yaml.load(f)

trains = np.load(params['path'])
ipts = np.load(params['ipt_path'])

n_samples = len(ipts[0])

start_t = time()
res_old, ids_old = keep_unique_weighted(trains, n_samples, params['sim_th'], params['min_len'])
end_t = time()
old_t = end_t - start_t

start_t = time()
res_new, ids_new = keep_unique_max(trains, n_samples, params['sim_th'], params['min_len'])
end_t = time()
new_t = end_t - start_t

print(f'All: {len(trains)}')
print(f'Kept (old): {len(res_old)} ({old_t:.3f}s) {[(i, len(x)) for i, x in zip(ids_old, res_old)]}')
print(f'Kept (new): {len(res_new)} ({new_t:.3f}s) {[(i, len(x)) for i, x in zip(ids_new, res_new)]}')

if params['plot']:
    ax1 = plt.subplot(211)
    ax1.eventplot(res_old, linelengths=0.8, linewidths=0.8)
    ax1.set_title('Old')
    ax1.axis('off')
    
    ax2 = plt.subplot(212)
    ax2.eventplot(res_new, linelengths=0.8, linewidths=0.8)
    ax2.set_title('New')
    ax2.axis('off')
    
    plt.show()