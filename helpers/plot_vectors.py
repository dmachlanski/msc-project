import numpy as np
import matplotlib.pyplot as plt

results = np.load('../experiments/eeg_sta/tmp/run25/kmckc_whitening_results.npy')

n_rows = 2
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, sharey=True)
axs = axs.flat

for r, ax in zip(results, axs):
    vector = r['sep'].flatten()
    ax.bar(range(vector.shape[0]), vector)

plt.show()