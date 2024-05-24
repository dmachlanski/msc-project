import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import get_peaks, get_cov_isi

path = '../experiments/eeg_sta/tmp/run29/emg_proc_s07_sess1_finger_results.npy'
peak_modes = ['auto', 'fixed', 'fixed_sq']
peaks_mode = peak_modes[2]
thr_mul = 3.0
fs = 4096
n_rows = 2
n_cols = 5
l_bound = 6
u_bound = 15

data = np.load(path)
all_peaks = get_peaks(data, peaks_mode, thr_mul)

covs = []
mdrs = []

for peaks in all_peaks:
    cov = get_cov_isi(peaks, True)
    covs.append(cov)

plt.figure(1)
plt.bar(range(len(covs)), covs)
plt.ylabel('CoV ISI')
plt.xlabel('Sources')
plt.axhline(np.mean(covs), color='red', linewidth=2)

#plt.figure(2)
fig, axs = plt.subplots(n_rows, n_cols, sharey=True)
axs = axs.flat
ax_id = 1
for i, (peaks, ax) in enumerate(zip(all_peaks, axs)):
    bin_sec = np.bincount(peaks//fs)
    bin_avg = np.mean(bin_sec)
    ax.bar(range(len(bin_sec)), bin_sec)
    ax.set_title(f'MU {i+1}')
    ax.set_xlabel('Time in [s]')
    ax.set_ylabel('Spikes per second')
    ax.axhline(bin_avg, color='green', linewidth=1)
    ax.fill_between(range(len(bin_sec)), l_bound, u_bound, color='green', alpha=0.1)

plt.show()