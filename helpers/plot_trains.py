import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, dest='data')
parser.add_argument('-l', type=int, dest='limit', default=0)

options = parser.parse_args()

data_path = None
if options.data:
    data_path = options.data
else:
    with open('plot_trains.yaml') as f:
        params = yaml.load(f)
        data_path = params['path']

mus = np.load(data_path)

to_plot = mus[:options.limit] if options.limit > 0 else mus

plt.figure(1)
plt.eventplot(mus, linelengths=0.8, linewidths=0.8, colors=['blue'])
plt.tight_layout()
plt.axis('off')
#plt.yticks(np.arange(0, 5), labels=['MU1', 'MU2', 'MU3', 'MU4', 'MU5'])
#plt.ylabel('Motor units')
#plt.xlabel('Time instants')

plt.figure(2)
shapes = [len(mu) for mu in mus]
shapes_mean = np.mean(shapes)
print(f'Mean spikes: {shapes_mean}')
plt.bar(range(len(shapes)), shapes)
plt.ylabel('Number of spikes')
plt.xlabel('Train index')
plt.axhline(shapes_mean, color='red', linewidth=2)

fig, axs = plt.subplots(1, len(mus), sharey=True)
ax_id = 1
for i, (mu, ax) in enumerate(zip(mus, axs)):
    bin_sec = np.bincount(mu//params['freq'])
    bin_avg = np.mean(bin_sec)
    ax.bar(range(len(bin_sec)), bin_sec)
    ax.set_title(f'MU {i+1}')
    ax.set_xlabel('Time in [s]')
    ax.set_ylabel('Spikes per second')
    ax.axhline(bin_avg, color='green', linewidth=1)
    ax.fill_between(range(len(bin_sec)), params['l_bound'], params['u_bound'], color='green', alpha=0.1)

plt.show()