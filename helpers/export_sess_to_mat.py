import numpy as np
from scipy.io import savemat

path = 'C:/Users/dmach/OneDrive - University of Essex/MSc/Experiments/eeg_sta/run46/emg_proc_s07_sess1_finger_'
max_components = 3
repetitions = 6
K = 16
decompose_vectors = False

test_load = f'{path}1_results.npy'
n_samples = len(np.load(test_load)[0]['sep'])
print(n_samples)

dim_0 = n_samples / (K+1) if decompose_vectors else n_samples
dim_0 = int(dim_0)
dim_1 = max_components * (K+1) if decompose_vectors else max_components
dim_1 = int(dim_1)

mat = np.zeros((dim_0, dim_1, repetitions))

for i in range(repetitions):
    results = np.load(f'{path}{i+1}_results.npy')
    for i_r, r in enumerate(results):
        if decompose_vectors:
            v_mixed = r['sep'].flatten()
            ids = np.arange(0, len(v_mixed), K+1)
            for lag in range(K+1):
                #print(ids + lag)
                mat[:, (i_r*(K+1))+lag, i] = v_mixed[ids + lag]
        else:
            mat[:, i_r, i] = r['sep'].flatten()

#savemat('../isctest/isctest_2.mat', dict(x = mat))
np.save('../isctest/isctest_1', mat)