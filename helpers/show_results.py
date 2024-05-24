import numpy as np

#path = '../experiments/eeg_sta/tmp/run23/emg_proc_s04_sess1_finger_1_results.npy'
path = 'C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/run44/run1/emg_proc_s07_sess1_finger_1_results.npy'
#path = 'C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/run43/run2/emg_noREF_proc_s07_sess1_finger_6_results.npy'

results = np.load(path)

for source in results:
    print(f"CoV={source['cov']:.3f}, Peaks={len(source['peaks'])}")