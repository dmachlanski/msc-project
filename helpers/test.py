import numpy as np

peaks = np.load('../data/sim/converted/test/S1_10MVC_peaks.npy')
print(peaks.shape)

for p in peaks:
    print(f'Shape = {p.shape}, [:10]: {p[:10]}')