import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import extend_data

data = np.load(f'../data/processed/emg_proc_s07_sess1_finger_1.npy')[[0,100]]
#data = data[:, :1000]

extended = extend_data(data, 2, 50)

fs = 4096
n_samples = extended.shape[-1]
seconds = n_samples / fs
x_ticks = np.linspace(0, seconds, n_samples)

fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
axs = axs.flat

for i, (ax, d) in enumerate(zip(axs, extended)):
    ax.plot(x_ticks, d)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$\mu$V')

plt.show()