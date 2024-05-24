import numpy as np
from scipy.io import loadmat

dir_path = '../data/sim/'
fs = 4096
#output = '../data/sim/converted/'
output = '../data/sim/converted/test3/'

cutoff = [20, 500]
order = 1

#subjects = np.arange(1, 16)
#mvcs = ['10', '30', '50'] * 5
subjects = [1]
mvcs = [10]

for s, mvc in zip(subjects, mvcs):
    filename = f'S{s}_{mvc}MVC'
    print(filename)
    m = loadmat(f'{dir_path}{filename}.mat', squeeze_me=True)
    print('\tPeaks')
    np.save(f'{output}{filename}_peaks', m['sFirings'])
    print('\tRecordings')
    recs = np.stack(m['sig_out'].reshape(90), axis=0)
    #filtered = butter_bandpass_filter(recs, cutoff[0], cutoff[1], fs, order)
    recs -= np.mean(recs, axis=0, keepdims=True)
    np.save(f'{output}{filename}_rec', recs)
