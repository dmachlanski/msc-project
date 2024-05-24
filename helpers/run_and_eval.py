import numpy as np
from kmckc import KmCKC
from time import time
from utils import merge_spikes, get_conf_mat

if __name__ == "__main__":
    print("--- START ---")
    print("")

    n_cells = 5
    electrodes = ['Neuronexus-32', 'Neuropixels-128']
    electrode = electrodes[0]
    t_sec = 40
    noise = 0
    path = f'rec_{n_cells}cells_{electrode}_{t_sec}sec_{noise}uV'

    x = np.load(f'/detop/temp/simulations/{path}/recordings.npy')

    start_t = time()
    alg = KmCKC(x, delays=10)
    end_t = time()
    print(f"Initialising: {end_t - start_t}s")

    start_t = time()
    mus = alg.decompose(Nmdl=10, r=10, Np=30, h=20, ipt_th=200, rk=200)
    end_t = time()
    print(f"Decomposition: {end_t - start_t}s")

    merged = merge_spikes(mus, 10, x.shape[1])

    np.save('last_run/mus', mus)
    np.save('last_run/merged', merged)
    print("Spikes merged and saved")

    trains = np.load(f'data/{path}/trains.npy', allow_pickle=True)
    threshold = 0.1
    neighbours = 2

    print("Evaluation:")

    for j, mu in enumerate(merged):
        if len(mu) < 1:
            continue
        for i, t in enumerate(trains):
            tp, fn, fp = get_conf_mat(t, mu, neighbours, x.shape[1])
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            if(recall > threshold or precision > threshold):
                print(f"Result({j}), MU({i}), Recall: {recall}, Prec: {precision}, TP: {tp}, FN: {fn}, FP: {fp}, Len diff: {abs(len(t)-len(mu))}")

    print("")
    print("--- END ---")