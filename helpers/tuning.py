import numpy as np
from kmckc import KmCKC

if __name__ == "__main__":
    #x = np.load('data/rec_5cells_Neuronexus-32_40sec_0uV/recordings.npy')
    x = np.load('/detop/temp/simulations/rec_5cells_Neuronexus-32_40sec_0uV/recordings.npy')
    #trains = np.load('data/rec_5cells_Neuronexus-32_40sec_0uV/trains_small.npy')

    alg = KmCKC(x, delays=9)

    n = 10
    rks = range(210, 310, 10)
    print(f"Data loaded. Starting {n} iterations.")

    for i in range(n):
        print(f"Run: {i+1}")
        #ipts, mus = alg.decompose(Nmdl=150, r=10, Np=10, h=60, ipt_th=200, similarily_th=0.9)
        mus = alg.decompose(Nmdl=150, r=10, Np=10, h=60, ipt_th=200, rk=rks[i], c=2)

        #np.save(f'tuning/ipts{i}', ipts)
        np.save(f'tuning/mus{i}', mus)