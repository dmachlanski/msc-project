import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=14)

# 125 channels as one was discarded in pre-processing
vector_orig = np.load('../data/decomposition/final/emg_proc_s04_sess1_finger_2_results.npy')[1]['sep'].flatten()

K_orig = 17
K_new = 4
n_channels = int(vector_orig.size/K_orig)
ids = np.arange(0, vector_orig.size, K_orig)

#plt.bar(np.arange(vector_orig.size), vector_orig)
#plt.show()

vectors = []
channels_to_show = 10


fig, axs = plt.subplots(1, 4, sharex=True, sharey=False)
axs = axs.flat

colors = ['blue', 'orange', 'green', 'red']
for lag, ax, color in zip(range(K_new), axs, colors):
    #vectors.append(vector_orig[ids + lag])
    v = vector_orig[ids + lag]
    x = np.arange(1, (channels_to_show * K_new)+1, K_new) + lag

    plt.figure(2)
    plt.bar(x, v[:channels_to_show], color=color, label=f'Lag = {lag}')
    plt.yticks([])
    plt.xlabel('Channels')
    plt.title(f'Complex vector (K = {K_new})')
    plt.legend()

    plt.figure(1)
    ax.bar(np.arange(1, channels_to_show+1), v[:channels_to_show], color=color)
    ax.set_yticks([])
    ax.set_xlabel('Channels')
    ax.set_title(f'Lag = {lag}')

plt.show()