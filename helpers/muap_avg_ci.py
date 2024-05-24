import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


rec = np.load('data/rec_5cells_Neuronexus-32_40sec_0uV/recordings.npy')
trains = np.load('data/rec_5cells_Neuronexus-32_40sec_0uV/trains.npy')

#r = rec[20]
t = trains[4]

# half before the spike and another half after the spike
window_half = 75

x = np.arange(((window_half * 2) + 1))

fig, axs = plt.subplots(4, 8, constrained_layout=True)
axs = trim_axs(axs, len(rec))
title_id=1

for ax, r in zip(axs, rec):
    data = np.zeros((len(t), ((window_half * 2) + 1)))

    for i, idx in enumerate(t):
        # Careful with out of range issue!
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