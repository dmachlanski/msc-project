import numpy as np
import sys
sys.path.append("../../")
from utils import get_peaks, eval_peaks_alt, get_cov_isi

print('---START---')

true_peaks = np.load('../../data/sim/converted/test3/S1_10MVC_peaks.npy')
results = np.load('S1_10MVC_results.npy')

#thresholds = np.arange(2.8, 3.3, 0.1)
thresholds = [2.9]
print(f'Thresholds to try: {thresholds}')

for thr in thresholds:
    print(f'Threshold = {thr:.2f}')
    peaks = get_peaks(results, 'fixed', thr)
    scores = eval_peaks_alt(true_peaks, peaks, len(results[0]['ipt']) + 100, 2, 0)
    if len(scores.keys()) > 0:
        covs = [get_cov_isi(peaks[scores[ks]['pred_id']], True) for ks in scores]
        print(f'CoVs: {covs} (mean = {np.mean(covs)})')
        print(f'Found {len(scores.keys())} MUs')
        print(scores)

print('---END---')