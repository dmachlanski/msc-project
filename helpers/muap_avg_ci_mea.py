import MEAutility as MEA
import numpy as np
import matplotlib.pylab as plt

nx = MEA.return_mea('Neuronexus-32')

rec = np.load('data/rec_5cells_Neuronexus-32_40sec_0uV/recordings.npy')
trains = np.load('data/rec_5cells_Neuronexus-32_40sec_0uV/trains.npy')

#r = rec[20]
t = trains[4]

# half before the spike and another half after the spike
window_half = 75
avgs = np.empty((32, 151))

for j, r in enumerate(rec):
    data = np.zeros((len(t), ((window_half * 2) + 1)))

    for i, idx in enumerate(t):
        # Careful with out of range issue!
        start = idx - window_half
        end = idx + window_half + 1
        data[i, :] = r[start:end]

    avg = np.mean(data, axis=0)
    avgs[j] = avg

MEA.plot_mea_recording(avgs, nx)