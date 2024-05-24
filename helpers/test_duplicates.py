import numpy as np

def spike_similarity(t1, t2, n, samples):
    longer_ids = None
    shorter = np.zeros((samples))
    counter = 0

    if len(t1) > len(t2):
        longer_ids = t1
        shorter[t2] = 1
    else:
        longer_ids = t2
        shorter[t1] = 1
    
    for id in longer_ids:
        start = max(0, id - n)
        end = min(samples, id + n + 1)
        if np.any(shorter[start:end]):
            counter += 1

    return counter/len(longer_ids)

unique = []
mus = np.load('last_run/merged.npy')

neighbours = 1
samples = 800000
l_sim = 0.9
s_sim = 0.9

for mu in mus:
    if len(unique) < 1:
        unique.append(mu)
        continue
    
    is_unique = True
    mu_length = len(mu)
    for d in unique:
        d_length = len(d)
        length_similarity = min(mu_length, d_length)/max(mu_length, d_length)

        if length_similarity < l_sim:
            continue
        elif spike_similarity(mu, d, neighbours, samples) >= s_sim:
            is_unique = False
            break

    if is_unique:
        unique.append(mu)

np.save('last_run/unique', unique)

print(f"Before: {len(mus)}, after: {len(unique)}")