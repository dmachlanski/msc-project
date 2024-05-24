import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('plot_peaks.yaml') as f:
    params = yaml.load(f)

source = np.load(params['source_path'])
trains = np.load(params['trains_path'])

channel = source[params['source_id']]
train = trains[params['train_id']]

plt.plot(channel)
plt.plot(train, channel[train], 'x')

plt.show()