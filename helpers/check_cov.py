import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append("../")
from utils import get_cov_isi

with open('check_cov.yaml') as f:
    params = yaml.load(f)

trains = np.load(params['path'])

covs = []

for t in trains:
    covs.append(get_cov_isi(t, True))

plt.bar(range(len(covs)), covs)
plt.show()