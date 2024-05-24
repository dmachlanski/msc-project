import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append("../")
from utils import get_cov_isi, find_peaks

np.set_printoptions(precision=2)

def get_covs(ipts, thr_mul, mode):
    covs = []
    for ipt in ipts:
        if mode == 'med':
            thr = thr_mul * np.median(np.abs(ipt) / 0.6745)
        else:
            thr = thr_mul * np.std(ipt)
        train, _ = find_peaks(ipt, height=thr)
        covs.append(get_cov_isi(train, False))
    return np.array(covs)

with open('test_thr_cov.yaml') as f:
    params = yaml.load(f)

ipts = np.load(params['path'])

if params['source_id'] < 0:
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for t in thresholds:
        tcov = get_covs(ipts, t, 'def')
        print(f'{t} --- {tcov} ({np.mean(tcov):.3f})')
else:
    ipt = ipts[params['source_id']]**2
    #ipt = ipt/np.max(ipt)
    peaks = find_peaks(ipt, height=np.std(ipt) * 8.0)[0]
    print(f'CoV = {get_cov_isi(peaks, True):.3f}')
    print(f'Number of peaks: {len(peaks)}')
    plt.plot(ipt)
    plt.plot(peaks, ipt[peaks], 'x')
    plt.show()