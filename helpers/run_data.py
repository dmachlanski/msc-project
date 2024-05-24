import numpy as np
from ckc import CKC
from kmckc import KmCKC
from time import time
from utils import merge_spikes

if __name__ == "__main__":

    data = [
        'rec_5cells_Neuronexus-32_40sec_0uV',
        'rec_5cells_Neuronexus-32_40sec_10uV',
        'rec_5cells_Neuronexus-32_40sec_20uV',
        'rec_5cells_Neuronexus-32_40sec_30uV',
        'rec_10cells_Neuronexus-32_40sec_0uV'
    ]

    for d in data:
        path = '/detop/temp/simulations/' + d + '/recordings.npy'
        x = np.load(path)

        start_t = time()
        alg = KmCKC(x, delays=9)
        end_t = time()
        print(f"Initialising: {end_t - start_t}s")

        start_t = time()
        mus = alg.decompose(Nmdl=150, r=10, Np=10, h=60, ipt_th=200)
        end_t = time()
        print(f"Decomposition: {end_t - start_t}s")

        merged = merge_spikes(mus, 10, x.shape[1])

        np.save(f'runs/mus_{d}', mus)
        np.save(f'runs/merged_{d}', merged)
