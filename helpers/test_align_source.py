import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import get_stas

recordings = np.load("../data/processed/emg_proc_s07_sess1_finger_6.npy")
peaks = np.load('C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/emg_proc/run46/emg_proc_s07_sess1_finger_6_results.npy')[0]['peaks']

n_rows = 14
n_cols = 9
window_half = 75

orig_stas = get_stas(recordings, peaks, window_half)
squared_stas = orig_stas**2

selected_observation_id = np.argmax(np.max(squared_stas, axis=1))
lag = np.argmax(squared_stas[selected_observation_id]) - window_half
print(f'Lag = {lag}')

aligned_stas = get_stas(recordings, peaks+lag, window_half)

fig, axs = plt.subplots(n_rows, n_cols, sharey=True)

for ax, orig_sta, aligned_sta in zip(axs.flat, orig_stas, aligned_stas):
    ax.axis('off')
    ax.plot(orig_sta, color='red')
    ax.plot(aligned_sta, color='blue')

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.show()