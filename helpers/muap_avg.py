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
t = trains[0]

# half before the spike and another half after the spike
window_half = 75

fig, axs = plt.subplots(4, 8, constrained_layout=True)
axs = trim_axs(axs, len(rec))
i=1

for ax, r in zip(axs, rec):
    avg = np.zeros(((window_half * 2) + 1))
    for idx in t:
        # Careful with out of range issue!
        start = idx - window_half
        end = idx + window_half + 1
        avg += r[start:end]
    avg /= len(t)
    ax.set_title(f'Rec:{i}')
    ax.plot(avg)
    i += 1

plt.show()