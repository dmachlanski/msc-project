import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import test_sep_vectors_alt, get_cov_isi

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

#path = '../experiments/eeg_sta/tmp/run23/emg_proc_s04_sess1_finger_1_results.npy'
#path = 'C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/run46/emg_proc_s07_sess1_finger_'
path = 'C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/final/emg_proc_s04_sess1_finger_'
ns = range(1, 7)
plot_vectors = False
plot_groups = False
show_bounds = True
norm_to_seconds = True
long_covs = True
decompose_vectors = True
plot_id = 1
# 0.3 or 0.5
threshold = 0.8
fs = 4096
K=16

merged_results = []
all_results = []
for n in ns:
    to_load = f'{path}{n}_results.npy'
    try:
        results = np.load(to_load)
    except:
        all_results.append([])
        continue
    all_vectors = []
    print(f'Rep {n}')
    for source in results:
        print(f"\tCoV={source['cov']:.3f}, Peaks={len(source['peaks'])}")
        if decompose_vectors:
            v_mixed = source['sep'].flatten()
            ids = np.arange(0, len(v_mixed), K+1)
            for lag in range(K+1):
                all_vectors.append(v_mixed[ids + lag])
        else:
            all_vectors.append(source['sep'].flatten())
    all_results.append(all_vectors)
    merged_results.append(results)

    if plot_vectors:
        if len(results) > 1:
            fig, axs = plt.subplots(1, len(results))
            for ax, result in zip(axs, results):
                vector = result['sep'].flatten()
                ax.bar(range(vector.shape[0]), vector)
        else:
            plt.figure(plot_id)
            vector = results[0]['sep'].flatten()
            plt.bar(range(vector.shape[0]), vector)
    plot_id += 1

grouped = []
for i, r_i in enumerate(all_results):
    for ii, r_ii in enumerate(r_i):
        for j, r_j in enumerate(all_results):
            if i >= j: continue
            for jj, r_jj in enumerate(r_j):
                score, lag = test_sep_vectors_alt(r_ii, r_jj, 'valid')
                if score > threshold:
                    if decompose_vectors:
                        add_group(grouped, (i+1, ((ii)//(K+1))+1), (j+1, ((jj)//(K+1))+1))
                    else:
                        print(f'Rep{i+1} MU{ii+1} - Rep{j+1} MU{jj+1} = {score:.3f} (lag: {lag})')
                        add_group(grouped, (i+1, ii+1), (j+1, jj+1))

sorted_groups = []
print('\nGroups')
for g in grouped:
    print(g)
    sorted_groups.append(sorted(list(g)))

print('\nSorted groups')
for sg in sorted_groups:
    print(sg)

if plot_groups:
    print()
    rep_len = len(merged_results[0][0]['ipt'])
    merged_mus = []
    merged_mus_covs = []
    for i, group in enumerate(sorted_groups):
        mu = np.array([])
        for tup in group:
            shifted_ids = (((tup[0]-1) * rep_len) + merged_results[tup[0] - 1][tup[1] - 1]['peaks'])
            mu = np.concatenate((mu, shifted_ids), axis=None)
        cov = get_cov_isi(mu, True)
        print(f'MU{i+1}: CoV={cov:.3f}, Peaks={len(mu)}')
        if norm_to_seconds:
            mu /= fs
        merged_mus.append(mu)
    if plot_vectors: plt.figure(plot_id)
    plt.eventplot(merged_mus, linelengths=0.8, linewidths=0.8)
    plt.tight_layout()
    plt.yticks(np.arange(0, len(merged_mus)), labels=[f'MU{i+1}' for i in range(len(merged_mus))])
    if show_bounds:
        for i in range(1, 6):
            plt.axvline(rep_len*i/fs if norm_to_seconds else rep_len*i, color='green', linestyle='dotted')
    plt.ylabel('Motor units', fontsize=16)
    plt.xlabel(f"Time {'[s]' if norm_to_seconds else 'instants'}", fontsize=16)

plt.show()