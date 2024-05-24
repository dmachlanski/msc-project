import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append("../")
from utils import get_cov_isi, find_peaks

with open('find_best_cov.yaml') as f:
    params = yaml.load(f)

ipt = np.load(params['path'])[params['source_id']]

covs = []
exclude = True

start_thr = best_thr = thr = np.std(ipt) * 2.0
best_cov = cov = get_cov_isi(find_peaks(ipt, height=thr)[0], exclude)

print(f'CoV = {cov:.2f} (start)')

step = params['alpha']
covs.append(cov)
for _ in range(params['max_iter']):
    thr = thr + step
    train = find_peaks(ipt, height=thr)[0]
    cov = get_cov_isi(train, exclude)
    covs.append(cov)
    #print(f'CoV = {cov:.2f}')
    if cov < best_cov and len(train) >= 10:
        best_thr = thr
        best_cov = cov

print(f'Thr = {best_thr:.2f}, CoV = {best_cov:.2f}')
print(f'Number of peaks: {len(find_peaks(ipt, height=best_thr)[0])}')

plt.figure(1)
plt.plot(ipt)
plt.axhline(start_thr, color='red')
plt.axhline(best_thr, color='green')

plt.figure(2)
plt.plot(covs)

plt.show()