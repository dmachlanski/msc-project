import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import test_sep_vectors_alt_3d, vec_to_3d

vector1 = np.load('C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/run46/emg_proc_s07_sess1_finger_4_results.npy')[0]['sep'].flatten()
vector2 = np.load('C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/run46/emg_proc_s07_sess1_finger_5_results.npy')[0]['sep'].flatten()

subject = 7
n_rows = 14
n_cols = 9

vector1_3d = vec_to_3d(vector1, subject)
vector2_3d = vec_to_3d(vector2, subject)

scores = test_sep_vectors_alt_3d(vector1_3d, vector2_3d, norm=True)
print(scores.shape)

fig, axs = plt.subplots(scores.shape[0], scores.shape[1], sharey=False)

for s_row, ax_row in zip(scores, axs):
    for s, ax in zip(s_row, ax_row):
        ax.bar(range(len(s)), s)
        ax.set_ylim(-1, 1)
        #ax.axis('off')

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.show()