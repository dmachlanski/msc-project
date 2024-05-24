import numpy as np
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import sys
sys.path.append("../")
from utils import get_cov_isi, sil_score_alt

np.set_printoptions(precision=2)

with open('test_sil_cov.yaml') as f:
    params = yaml.load(f)

ipt = np.load(params['path'])[params['source_id']]

if params['power']: ipt = ipt**2

all_peaks, _ = find_peaks(ipt, height=0)
candidate = ipt[all_peaks].reshape(-1, 1)
kmeans = KMeans(n_clusters=2).fit(candidate)
score = silhouette_score(candidate, kmeans.labels_)

c_high = np.argmax(kmeans.cluster_centers_)
local_ids = np.where(kmeans.labels_ == c_high)[0]
peaks = all_peaks[local_ids]

#print(kmeans.cluster_centers_)
#print(np.max(kmeans.cluster_centers_))
#print(np.argmax(kmeans.cluster_centers_))
#print(f'Length of c0: {len(np.where(kmeans.labels_ == 0)[0])}')
#print(f'Length of c1: {len(np.where(kmeans.labels_ == 1)[0])}')

print(f'SIL = {score:.3f}')
print(f"CoV = {get_cov_isi(peaks, params['exclude']):.3f}")
print(f'Number of peaks: {len(peaks)}')

std_peaks = find_peaks(ipt, height=np.std(ipt) * 6.0)[0]
print(f"(STD) CoV = {get_cov_isi(std_peaks, params['exclude']):.3f}")
print(f'(STD) Number of peaks: {len(std_peaks)}')

plt.plot(ipt)
plt.plot(peaks, ipt[peaks], 'x')
plt.show()