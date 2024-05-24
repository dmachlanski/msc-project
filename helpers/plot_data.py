import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('plot_data.yaml') as f:
    params = yaml.load(f)

data = np.load(params['path'])[0]

plt.plot(data.T)
plt.axis('off')
plt.show()