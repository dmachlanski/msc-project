import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
import yaml

with open('test_corr.yaml') as f:
    params = yaml.load(f)

ipts = np.load(params['path'])
ipt1 = ipts[params['ids'][0]]
ipt2 = ipts[params['ids'][1]]

if params['norm']:
    ipt1 = ipt1 / np.max(ipt1)
    ipt2 = ipt2 / np.max(ipt2)

if params['filter']:
    ipt1_th = np.std(ipt1) * 3.0
    ipt2_th = np.std(ipt2) * 3.0
    ipt1[ipt1 < ipt1_th] = 0
    ipt2[ipt2 < ipt2_th] = 0

if params['to_peaks']:
    peaks1, _ = find_peaks(ipt1, np.std(ipt1) * params['x_std'])
    peaks2, _ = find_peaks(ipt2, np.std(ipt2) * params['x_std'])
    x1 = np.zeros(len(ipt1))
    x2 = np.zeros(len(ipt2))
    x1[peaks1] = 1
    x2[peaks2] = 1
    if params['n'] > 0:
        for i in range(1, params['n'] + 1):
            x2[peaks2 - i] = 1
            x2[peaks2 + i] = 1
else:
    x1 = ipt1
    x2 = ipt2

if params['coef']:
    corr_coef = np.corrcoef(x1, x2)
    print(f'R = {corr_coef}')

#corr_base1 = correlate(x1, x1, 'valid')[0]
#corr_base2 = correlate(x2, x2, 'valid')[0]
#print(f'Base corr = {corr_base:.2f}')
corr_base1 = len(peaks1)
corr_base2 = len(peaks2)

corr = np.max(correlate(x1, x2, 'same'))
#lag = np.argmax(corr)-corr.size//2
#print(corr)

#plt.plot(np.arange(corr.size) - corr.size/2, corr)
#plt.show()

similarity = corr / np.sqrt(corr_base1 * corr_base2)

#print(f'Max corr = {max_corr:.2f}, lag = {lag}')
print(f'Similarity ratio = {similarity:.3f}')

if params['plot']:
    if params['to_peaks']:
        plt.eventplot([peaks1, peaks2], linelengths=0.8, linewidths=0.8)
        plt.tight_layout()
    else:
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(x1)
        axs[1].plot(x2)
        fig.tight_layout()

    plt.show()