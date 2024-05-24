import numpy as np
import elephant.conversion as conv

trains = np.load('data/rec_5cells_Neuronexus-32_60sec_0uV/spiketrains.npy', allow_pickle=True)
result = []

for t in trains:
    x = conv.BinnedSpikeTrain(t, num_bins=1200000)
    result.append(x.spike_indices[0])

np.save('data/rec_5cells_Neuronexus-32_60sec_0uV/trains', result)