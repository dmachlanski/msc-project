import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append("../")
from utils import extend_data

with open('test_whiten.yaml') as f:
    params = yaml.load(f)

data_orig = np.load(params['path'])

# remove noisy channel
data = np.delete(data_orig, 116, 0)
#data = data_orig
#extend
x = extend_data(data, 1)
#x = data

#subtract the mean
x = x - np.mean(x, axis=1, keepdims=True)

Cxx = np.matmul(x, x.T)/x.shape[1]

plt.figure(1)
plt.matshow(Cxx, fignum=1)
plt.colorbar().ax.tick_params(labelsize=14)

#whiten
d, V = np.linalg.eigh(Cxx)

#regularisation
l = len(d)//2
ids = np.argsort(d)[:l]
reg = np.mean(d[ids])
#d[d < m] = 0

#reg = 1e-18

D = np.diag(1. / np.sqrt(d + reg))
W = np.dot(np.dot(V, D), V.T)
x_white = np.dot(W, x)

#plt.figure(2)
#plt.matshow(W, fignum=2)
#plt.colorbar().ax.tick_params(labelsize=14)

Czz = np.matmul(x_white, x_white.T)/x_white.shape[1]
plt.figure(2)
plt.matshow(Czz, fignum=2)
plt.colorbar().ax.tick_params(labelsize=14)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)

fs = 4096
x_ticks_orig = np.linspace(0, data_orig[0].shape[-1] / fs, data_orig[0].shape[-1])
x_ticks_whiten = np.linspace(0, x[0].shape[-1] / fs, x[0].shape[-1])

axs[0].plot(x_ticks_orig, data_orig[10])
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel(r'$\mu$V')

axs[1].plot(x_ticks_whiten, x_white[10])
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel(r'$\mu$V')

plt.show()