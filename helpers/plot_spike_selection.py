import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

results = np.load(f'C://Users//dmach//OneDrive - University of Essex//MSc//Experiments//emg_proc//final//emg_proc_s04_sess1_finger_2_results.npy')

ipt = results[0]['ipt']
ipt_sq = ipt ** 2
thr = np.std(ipt_sq) * 3.0
peaks = find_peaks(ipt_sq, height=thr)[0]

fs = 4096
n_samples = ipt.shape[-1]
seconds = n_samples / fs
x_ticks = np.linspace(0, seconds, n_samples)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)

axs[0].plot(x_ticks, ipt)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel(r'$\mu$V')

axs[1].plot(x_ticks, ipt_sq)
axs[1].axhline(thr, color='red', label='threshold')
axs[1].plot(peaks / fs, ipt_sq[peaks], 'x', label='selected peaks')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel(r'$\mu$V')
axs[1].legend()

plt.show()