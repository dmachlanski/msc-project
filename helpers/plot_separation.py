import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append("../")
from utils import vec_to_3d

BAD_CHANNELS = {4: [54],
                5: [49],
                6: [44, 53, 60, 62, 71, 80, 89, 95, 98],
                7: [116],
                8: [95, 116]}

with open('plot_separation.yaml') as f:
    params = yaml.load(f)

vector = np.load(params['path'])[params['id']]['sep'].flatten()

# TODO
if params['norm']:
    vector = ((vector - np.min(vector))/(np.max(vector) - np.min(vector))) - 0.5

# From utils
#a = (v1 - np.mean(v1)) / (np.std(v1) * len(v1))
#b = (v2 - np.mean(v2)) / (np.std(v2))
#return np.max(correlate(a, b, 'same'))

plt.figure(1)
plt.bar(range(vector.shape[0]), vector)

vector3d = vec_to_3d(vector, params['subject'], params['n_rows'], params['n_cols'], params['extension_factor'])
print(vector3d.shape)
print(np.mean(vector3d, axis=2, keepdims=True).shape)
print(np.std(vector3d, axis=2, keepdims=True).shape)
print(vector3d.shape[2])

fig, axs = plt.subplots(params['n_rows'], params['n_cols'])

for v_row, ax_row in zip(vector3d, axs):
    for v, ax in zip(v_row, ax_row):
        ax.bar(range(len(v)), v)
        ax.axis('off')

plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
plt.show()