import numpy as np
import matplotlib.pyplot as plt
import yaml

with open('plot_ipts.yaml') as f:
    params = yaml.load(f)


x = np.load(params['path'])
#s_n0 = np.load('tmp/s_n0.npy').flatten()

fs = 4096
n_samples = len(x[-1]['ipt']) if params['new_format'] else len(x[-1])
seconds = n_samples / fs
x_ticks = np.linspace(0, seconds, n_samples)

if params['amount'] > 1:
    fig, axs = plt.subplots(params['amount'])

    for i, xi in enumerate(range(params['start'], params['start'] + params['amount'])):
        if params['new_format']:
            axs[i].plot(x[xi]['ipt']**2)
        else:
            axs[i].plot(x[xi]**2)

    fig.tight_layout()
else:
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\mu$V')
    if params['new_format']:
        plt.plot(x_ticks, x[params['start']]['ipt'])
    else:
        plt.plot(x_ticks, x[params['start']])

plt.show()