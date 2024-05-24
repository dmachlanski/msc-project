import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import yaml

def ipt_to_spikes(ipt, x_std, dist):
    mu, _ = find_peaks(ipt, height=np.std(ipt) * x_std, distance=dist)
    return mu

with open('test_sil.yaml') as f:
    params = yaml.load(f)

ipts = np.load(params['path'])
ipt = ipts[params['ipt_id']]
#ipt = ipt**2
train = ipt_to_spikes(ipt, params['std_dev_size'], params['spike_dist'])
candidate = ipt[train].reshape(-1, 1)

kmeans = KMeans(n_clusters=2).fit(candidate)

#candidate = x[:, s]**2
sample_size = 35000 if candidate.shape[0] > 35000 else candidate.shape[0]

#print(f'Sample size: {sample_size}')

sil = silhouette_score(candidate, kmeans.labels_, sample_size=sample_size)
print(f'SIL = {sil:.3f}')

plt.plot(ipt)
plt.plot(train, ipt[train], 'x')
plt.show()