import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from utils import get_peaks, get_cov_isi, quality_filter_alt, duplicate_filter

#path = '../experiments/eeg_sta/tmp/run25/kmckc_no_whitening.npy'
#path = '../experiments/eeg_sta/tmp/run25/kmckc_whitening.npy'
path = '../experiments/eeg_sta/tmp/run29/emg_proc_s07_sess1_finger_results.npy'
peak_modes = ['auto', 'fixed', 'fixed_sq']
peaks_mode = peak_modes[2]
thr_mul = 3.0
cov_thr = 0.3
modes = ['train', 'vector', 'vector_alt']
compare_mode = modes[0]
compare_thr = 0.1
skip_plots = True
save_results = True

data = np.load(path)

#plt.plot(data[142]['ipt'])
#plt.show()
#quit(0)

all_peaks = get_peaks(data, peaks_mode, thr_mul)

covs = []
lens = []
quality_covs = []

for i, peaks in enumerate(all_peaks):
    cov = get_cov_isi(peaks, True)
    covs.append(cov)
    lens.append(len(peaks))
    if cov < cov_thr:
        quality_covs.append(cov)

quality_data = quality_filter_alt(data, cov_thr, 10, peaks_mode, thr_mul, 4096)
unique_data = duplicate_filter(quality_data, len(data[0]['ipt']), compare_thr, compare_mode)

print(f'{len(unique_data)} unique sources found')

if save_results:
    np.save('../experiments/eeg_sta/tmp/run25/kmckc_whitening_results', unique_data)

for u in unique_data:
    print(f"CoV = {u['cov']:.3f}, Peaks = {len(u['peaks'])}")

if skip_plots: quit(0)

plt.figure(1)
plt.bar(range(len(covs)), covs)
plt.ylabel('CoV ISI')
plt.xlabel('Sources')
plt.axhline(np.mean(covs), color='red', linewidth=2)

plt.figure(2)
plt.bar(range(len(lens)), lens)
plt.ylabel('Number of spikes')
plt.xlabel('Sources')

plt.figure(3)
plt.bar(range(len(quality_covs)), quality_covs)
plt.ylabel('CoV ISI')
plt.xlabel('Sources')
plt.axhline(np.mean(quality_covs), color='red', linewidth=2)

plt.figure(4)
plt.hist(quality_covs, bins=[0.0,0.1,0.2,0.3,0.4,0.5], rwidth=0.9)
plt.ylim(0, 40)

plt.show()