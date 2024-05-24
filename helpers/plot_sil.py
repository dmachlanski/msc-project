import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append("../")
from utils import sil_score_alt

with open('plot_sil.yaml') as f:
    params = yaml.load(f)

ipts = np.load(params['ipts'])
#trains = np.load(params['trains'])

scores = []
clusters = []

for ipt in ipts:
    sil, km = sil_score_alt(ipt**2)
    scores.append(sil)
    clusters.append(km.cluster_centers_)

print([f'{s:.3f}' for s in scores])

for c in clusters:
    print(c)

plt.bar(range(len(scores)), scores)
plt.ylabel('SIL')
plt.xlabel('Candidates')
plt.ylim(0, 1)

plt.show()