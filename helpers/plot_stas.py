import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

'''
Perform phase-locked averaging
- Test with ground truth spikes first to ensure correctness.
'''
def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

with open('plot_stas.yaml') as f:
    params = yaml.load(f)

"""
rec = None
for i in range(6):
    x = np.load(f"{params['rec_path']}{i+1}.npy")
    rec = np.hstack((rec, x)) if i > 0 else x
"""

rec = np.load(f"{params['rec_path']}")
#rec = np.load(f"{params['rec_path']}")[:, ::2]
trains = np.load(params['train_path'])

rs = rec[params['ch_start'] : params['ch_end']]
t = trains[params['mu_id']]

#print(rs.shape)
#print(t.shape)
#print(len(t))

# half before the spike and another half after the spike
window_half = params['window_half']

x = np.arange(((window_half * 2) + 1))

fig, axs = plt.subplots(params['n_rows'], params['n_cols'], constrained_layout=True)
axs = trim_axs(axs, len(rs))
title_id=1

for ax, r in zip(axs, rs):
    data = np.zeros((len(t), ((window_half * 2) + 1)))

    for i, idx in enumerate(t):
        # Careful with out of range issue!
        if idx < window_half or idx+window_half > len(r): continue

        start = idx - window_half
        end = idx + window_half + 1

        data[i, :] = r[start:end]

    avg = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)
    upper_bound = avg + stdev
    lower_bound = avg - stdev

    ax.set_title(f'Rec:{title_id}')
    ax.plot(avg, color='blue')
    ax.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.4)
    title_id += 1

plt.show()