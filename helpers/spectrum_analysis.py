import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.signal import welch

with open('spectrum_analysis.yaml') as f:
    params = yaml.load(f)

fig1, axs1 = plt.subplots(params['n_rows'], params['n_cols'], sharey=False)
axs1 = axs1.flat

fig2, axs2 = plt.subplots(params['n_rows'], params['n_cols'], sharey=False)
axs2 = axs2.flat

data = np.load(params['path'])

#take = params['freq'] * 5
#data = data[:, -take:]

for i, (ax1, ax2, d) in enumerate(zip(axs1, axs2, data)):
    f, Pxx_spec = welch(d, params['freq'], 'hann', params['window'], scaling='spectrum')
    ax1.axis('off')
    ax1.set_title(i, fontsize='medium', loc='center', pad=-10)
    #ax1.semilogy(f, np.sqrt(Pxx_spec))
    ax1.plot(f, np.sqrt(Pxx_spec))

    ax2.axis('off')
    ax2.set_title(i, fontsize='medium', loc='center', pad=0)
    ax2.plot(d)

fig1.subplots_adjust(0.01, 0.01, 0.99, 0.99)
fig2.subplots_adjust(0.01, 0.01, 0.99, 0.99)
plt.show()