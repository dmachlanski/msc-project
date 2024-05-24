"""
This file contains many important utility methods (evaluation metrics, whitening, data extension, etc.). Too many to list them all here.
"""

import numpy as np
from scipy.signal import find_peaks, correlate
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Bad channels identified in the experimental HDsEMG recordings.
BAD_CHANNELS = {4: [54],
                5: [49],
                6: [44, 53, 60, 62, 71, 80, 89, 95, 98],
                7: [116],
                8: [95, 116]}

def get_stas(observations, peaks, window_half):
    """
    Compute the spike-triggered averages given the observations,
    a spike train and half of the epoch length in samples.
    """
    results = []
    for o in observations:
        data = np.zeros((len(peaks), ((window_half * 2) + 1)))
        for i, p in enumerate(peaks):
            if p < window_half or p+window_half > len(o): continue
            start = p - window_half
            end = p + window_half + 1
            data[i, :] = o[start:end]
        results.append(np.mean(data, axis=0))
    return np.array(results)

def get_source_lag(observations, peaks, window_half):
    """
    Get the time lag required for source alignment.
    """
    orig_stas = get_stas(observations, peaks, window_half)
    squared_stas = orig_stas**2
    selected_observation_id = np.argmax(np.max(squared_stas, axis=1))
    return np.argmax(squared_stas[selected_observation_id]) - window_half

def align_peaks(observations, peaks, window_half):
    """
    Get the time lag and apply the alignment to the given spike train.
    """
    lag = get_source_lag(observations, peaks, window_half)
    return peaks + lag

def vec_to_3d(vector, subject, n_rows=14, n_cols=9, ext_factor=16):
    n_channels = n_rows * n_cols
    vector2d = np.zeros((n_channels, ext_factor + 1))
    vector_start_id = 0
    for i in range(n_channels):
        if i not in BAD_CHANNELS[subject]:
            end_id = vector_start_id + ext_factor + 1
            vector2d[i] = vector[vector_start_id : end_id]
            vector_start_id += ext_factor + 1
    return vector2d.reshape((n_rows, n_cols, ext_factor + 1))

def eval_peaks(actual, pred, samples, n=0, lag=5, verbose=0):
    """
    Evaluate a spike train given we have access to the gold standard.
    """
    grouped = {}
    for i, p in enumerate(pred):
        for j, a in enumerate(actual):
            p_arr = np.array(p)
            a_arr = np.array(a)
            best_score = {'roa': 0.0}
            for l in range(-lag, lag, 1):    
                scores = get_eval(a_arr, p_arr + l, samples, n)
                if scores['roa'] > best_score['roa']:
                    best_score = scores
            if best_score['roa'] < 0.3: continue
            if j in grouped and best_score['roa'] <= grouped[j]['roa']: continue
            best_score['actual_id'] = j
            best_score['pred_id'] = i
            if verbose > 1: print('Adding quality source!')
            grouped[j] = best_score
    return grouped

def eval_peaks_alt(actual, pred, samples, fs=4096., n=0, verbose=0):
    """
    Alternative approach to the evaluation of spike trains given the gold standard access.
    """
    grouped = {}
    for i, p in enumerate(pred):
        for j, a in enumerate(actual):
            peak_max = max(np.max(p), np.max(a))
            score = get_eval_corr(a, p, peak_max + 1 + n, n)
            if score['roa'] < 0.3: continue
            if verbose > 1: print('RoA > 0.3')
            if j in grouped and score['roa'] <= grouped[j]['roa']: continue
            score['actual_id'] = j
            score['pred_id'] = i
            score['cov'] = get_cov_isi(p, exclude=True, fs=fs)
            score['mdr'] = len(p) / (samples / fs)
            if verbose > 1: print('Adding quality source!')
            grouped[j] = score
    return grouped

def get_eval_corr(x1, x2, n_samples, n):
    """
    Computes evaluation metrics via cross-correlation.
    Much faster than manual counting of TPs, FPs and FNs.
    And checks other time lags too!
    """
    t1 = np.zeros(n_samples)
    t2 = np.zeros(n_samples)
    t1[x1] = 1
    t2[x2] = 1

    if n > 0:
        for i in range(1, n + 1):
            t2[x2 - i] = 1
            t2[x2 + i] = 1

    tp = np.max(correlate(t1, t2, 'same'))
    fn = len(x1) - tp
    fp = len(x2) - tp
    roa = tp/(tp+fn+fp)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    return {'roa': roa, 'recall': recall, 'precision': precision}

def get_peaks_single(ipt):
    """
    Extracts candidate peaks from IPTs via clustering.
    """
    x = ipt**2
    all_peaks, _ = find_peaks(x, height=0)
    candidate = x[all_peaks].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(candidate)
    c_high = np.argmax(kmeans.cluster_centers_)
    local_ids = np.where(kmeans.labels_ == c_high)[0]
    return all_peaks[local_ids]

def get_peaks(data, mode='auto', thr_mul=3.0):
    """
    An umbrella method for different strategies to peak extraction from IPTs.
    """
    results = []
    for d in data:
        if mode == 'auto':
            x = d['ipt']**2
            all_peaks, _ = find_peaks(x, height=0)
            candidate = x[all_peaks].reshape(-1, 1)
            kmeans = KMeans(n_clusters=2).fit(candidate)
            c_high = np.argmax(kmeans.cluster_centers_)
            local_ids = np.where(kmeans.labels_ == c_high)[0]
            peaks = all_peaks[local_ids]
        elif mode == 'fixed':
            peaks = find_peaks(d['ipt'], height=np.std(d['ipt']) * thr_mul)[0]
        elif mode == 'fixed_sq':
            x = d['ipt']**2
            peaks = find_peaks(x, height=np.std(x) * thr_mul)[0]
        else:
            print('Unknown get_peaks mode')
            return None
        results.append(peaks)
    return results

def quality_filter(data, sil, cov, min_len, verbose=0):
    """
    One of the first versions of the quality filter.
    This one uses SIL score (very slow) and CoV.
    """
    to_keep = []
    for d in data:
        x = d['ipt']**2
        all_peaks, _ = find_peaks(x, height=0)
        candidate = x[all_peaks].reshape(-1, 1)
        kmeans = KMeans(n_clusters=2).fit(candidate)

        c_high = np.argmax(kmeans.cluster_centers_)
        local_ids = np.where(kmeans.labels_ == c_high)[0]
        peaks = all_peaks[local_ids]

        if len(peaks) < min_len: continue

        cov_score = get_cov_isi(peaks, True)
        if cov_score > cov: continue

        sil_score = silhouette_score(candidate, kmeans.labels_)
        if sil_score < sil: continue
        
        d['peaks'] = peaks
        d['cov'] = cov_score
        d['sil'] = sil_score
        to_keep.append(d)
    return to_keep

def quality_filter_alt(data, cov, min_len, mode, thr_mul, fs, verbose=0):
    """
    This method extracts peaks from IPTs and calculates quality metrics (CoV and MDR).
    Keeps only the sources satisfying defined criteria.
    """
    to_keep = []
    for d in data:
        if mode == 'auto':
            x = d['ipt']**2
            all_peaks, _ = find_peaks(x, height=0)
            candidate = x[all_peaks].reshape(-1, 1)
            kmeans = KMeans(n_clusters=2).fit(candidate)
            c_high = np.argmax(kmeans.cluster_centers_)
            local_ids = np.where(kmeans.labels_ == c_high)[0]
            peaks = all_peaks[local_ids]
        elif mode == 'fixed':
            peaks = find_peaks(d['ipt'], height=np.std(d['ipt']) * thr_mul)[0]
        elif mode == 'fixed_sq':
            x = d['ipt']**2
            peaks = find_peaks(x, height=np.std(x) * thr_mul)[0]
        else:
            print('Unknown get_peaks mode')
            return None

        if len(peaks) < min_len: continue

        # Keep mean discharge rate within [6; 40]
        # Unfortunately hardcoded. Should be converted to parameters.
        mdr = len(peaks) / (len(d['ipt']) / fs)
        if mdr < 6 or mdr > 40: continue

        cov_score = get_cov_isi(peaks, True)
        if cov_score > cov: continue
        
        d['peaks'] = peaks
        d['cov'] = cov_score
        to_keep.append(d)
    return to_keep

def duplicate_filter(data, n_samples, corr_th, mode):
    """
    Filters out duplicated sources - returns only the unique ones.
    """
    if mode not in {'train', 'vector', 'vector_alt'}:
        print('Unregocnised filtering mode')
        return

    unique = {}
    # Start with the longest train
    longest = data[0]
    longest_id = 0
    for id, d in enumerate(data):
        if len(longest['peaks']) < len(d['peaks']):
            longest = d
            longest_id = id
    unique[longest_id] = longest

    for i, d in enumerate(data):
        is_unique = True
        for k in unique:
            if (mode == 'train' and test_similarity(d['peaks'], unique[k]['peaks'], n_samples) >= corr_th) or (mode == 'vector' and test_sep_vectors(d['sep'], unique[k]['sep']) >= corr_th) or (mode == 'vector_alt' and test_sep_vectors_alt(d['sep'], unique[k]['sep']) >= corr_th):
                is_unique = False
                break

        if is_unique:
            unique[i] = d
        elif d['cov'] < unique[k]['cov']:
            unique.pop(k)
            unique[i] = d

    return list(unique.values())    

def extend_data(data, k, step=1):
    """
    Extends the observations by the 'k' factor.
    Also accepts a custom 'step'.
    """
    extended = np.empty(((k+1)*data.shape[0], data.shape[1] + (k*step)))
    index = 0
    for d in range(data.shape[0]):
        for i in range(k+1):
            extended[index] = np.pad(data[d], (i*step, (k-i)*step), 'constant')
            index += 1
    return extended

def whiten_data(data):
    """
    Performs data whitening (based on eigenvalue decomposition).
    """
    # Assuming channels x samples
    x = data - np.mean(data, axis=1, keepdims=True)
    Cxx = np.matmul(x, x.T)/x.shape[1]
    #reg = 1e-18
    d, V = np.linalg.eigh(Cxx)
    l = len(d)//2
    ids = np.argsort(d)[:l]
    reg = np.mean(d[ids])
    D = np.diag(1. / np.sqrt(d + reg))
    W = np.dot(np.dot(V, D), V.T)
    return np.dot(W, x), reg

def find_highest_peaks(x, count, height, dist):
    """
    Selects only 'count' highest peaks above provided threshold.
    """
    peaks, _ = find_peaks(x, height=height, distance=dist)
    d = dict(zip(peaks, x[peaks]))
    return np.array(sorted(d, key=d.get, reverse=True)[:count])

def sil_score(x, x_std=3.0):
    """
    Calculates the SIL score for a given source.
    """
    peaks, _ = find_peaks(x, height=np.std(x) * x_std)
    candidate = x[peaks].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(candidate)
    return silhouette_score(candidate, kmeans.labels_)

def sil_score_alt(x):
    """
    Alternative version to the 'sil_score' function.
    """
    peaks, _ = find_peaks(x, height=0)
    candidate = x[peaks].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(candidate)
    score = silhouette_score(candidate, kmeans.labels_)
    return score, kmeans

def print_sil_cov(x, exclude):
    """
    A helper method that calculates and prints the SIL score for a given source.
    """
    x = x**2
    all_peaks, _ = find_peaks(x, height=0)
    candidate = x[all_peaks].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2).fit(candidate)
    score = silhouette_score(candidate, kmeans.labels_)
    c_high = np.argmax(kmeans.cluster_centers_)
    local_ids = np.where(kmeans.labels_ == c_high)[0]
    peaks = all_peaks[local_ids]

    print("SIL = {}, CoV = {}, Number of peaks: {}".format(score, get_cov_isi(peaks, exclude), len(peaks)))

def sil_filter(x, labels, threshold=0.8):
    # Code provided by Dr Ana Matran-Fernandez
    keep_these = []
    for s in range(x.shape[-1]):
        candidate = x[:, s]**2
        sample_size = 35000 if candidate.shape[0] > 35000 else candidate.shape[0]
        sil = silhouette_score(candidate, labels[s], sample_size=sample_size)
        if sil >= threshold:
            keep_these.append(s)
    return keep_these

def best_recall(actuals, pred, samples, n=0):
    """
    Returns the best Recall value obtained.
    """
    results = []
    for a in actuals:
        r = get_recall(a, pred, samples, n)
        results.append(r)
    return max(results)

def get_eval(actual, pred, samples, n=0):
    """
    Calculate the evaluation metrics for the gold standard case (this is the slow version).
    """
    tp, fn, fp = get_conf_mat(actual, pred, n, samples)
    roa = tp/(tp+fn+fp)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    return {'roa': roa, 'recall': recall, 'precision': precision}

def get_recall(actual, pred, samples, n=0):
    """
    Gets the Recall (a.k.a. sensitivity) metric.
    """
    tp, fn = _get_tpfn(actual, pred, n, samples)
    return tp/(tp+fn)

def get_precision(actual, pred, samples, n=0):
    """
    Gets the precision metric.
    """
    tp, fp = _get_tpfp(actual, pred, n, samples)
    return tp/(tp+fp)

def get_conf_mat(actual, pred, n=0, samples=100000):
    """
    Count the TPs, FPs and FNs.
    """
    tp, fn = _get_tpfn(actual, pred, n, samples)
    _, fp = _get_tpfp(actual, pred, n, samples)
    return (tp, fn, fp)

def merge_spikes(mus, window, samples):
    """
    Merges spikes that are very close to each other.
    Mostly to reduce false positive rates.
    """
    result = []
    for mu in mus:
        pred_t = np.zeros((samples))
        res = []
        pred_t[mu] = 1
        j=0
        while(j < len(mu)):
            start = mu[j]
            end = min(mu[j] + window + 1, samples)
            count = int(sum(pred_t[start:end]))
            if count > 1:
                r = np.where(pred_t[start:end] == 1)[0]
                # shift by mu[j] as 'r' indexes are local
                avg = round(sum(r + mu[j]) / len(r))
            else:
                avg = mu[j]
            res.append(int(avg))
            j += count
        result.append(np.array(res))
    return result

def get_cov_isi(x, exclude=False, fs=4096., tmin=20., tmax=250.):
    """
    Calculates CoV ISI of a given train.
    Does not count abnormally short or long intervals.
    """
    if len(x) < 2: return 1

    isi = [x[i+1] - x[i] for i in range(len(x) - 1)]

    if exclude:
        min_isi = (tmin / 1000) * fs
        max_isi = (tmax / 1000) * fs
        isi = [i for i in isi if min_isi < i < max_isi]

    if len(isi) < 2: return 1

    return np.std(isi) / np.mean(isi)

def get_pnr(ipt, thr):
    """
    Calculates the pulse-to-noise ratio metric.
    """
    all_peaks, _ = find_peaks(ipt, height=0)
    high_peaks, _ = find_peaks(ipt, height=thr)
    low_peaks = np.array(list(set(all_peaks).difference(set(high_peaks))))
    num = np.mean(ipt[high_peaks]**2)
    denom = np.mean(ipt[low_peaks]**2)
    return 10 * np.log10(num / denom)

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

def test_sep_vectors(v1, v2, norm=True):
    if norm:
        return np.sum(v1*v2) / np.sqrt(np.sum(v1**2) * np.sum(v2**2))
    else:
        return np.sum(v1*v2)

def test_sep_vectors_alt(v1, v2, mode='same'):
    a = (v1 - np.mean(v1)) / (np.std(v1) * len(v1))
    b = (v2 - np.mean(v2)) / (np.std(v2))
    scores = correlate(a, b, mode)
    arg = np.argmax(scores)
    lag = arg - (len(scores)//2)
    return scores[arg], lag

def test_sep_vectors_alt_3d(v1, v2, norm=False):
    if norm:
        fudge = 1e-15
        a = (v1 - np.mean(v1, axis=None, keepdims=True)) / ((np.std(v1, axis=None, keepdims=True)) + fudge)
        b = (v2 - np.mean(v2, axis=None, keepdims=True)) / ((np.std(v2, axis=None, keepdims=True)) + fudge)
        a /= a.size
        return correlate(a, b, 'same')
    else:
        return correlate(v1, v2, 'same')

def test_similarity(x1, x2, n_samples, mode='weighted', n=0):
    if mode == 'weighted':
        return _correlate_weighted(x1, x2, n_samples)
    elif mode == 'simple':
        return _correlate_simple(x1, x2, n_samples, n)
    elif mode == 'locked':
        return _correlate_locked(x1, x2, n_samples, n)
    else:
        print('Unrecognised mode')
        return -1

def _correlate_locked(x1, x2, n_samples, n):
    longer = np.zeros((n_samples))
    shorter = np.zeros((n_samples))
    if len(x1) > len(x2):
        longer_ids = x1
        shorter_ids = x2
    else:
        longer_ids = x2
        shorter_ids = x1
    shorter[shorter_ids] = 1
    longer[longer_ids] = 1
    for i in range(1, n+1):
        longer[longer_ids - i] = 1
        longer[longer_ids + i] = 1
    return np.sum(np.multiply(shorter, longer)) / len(shorter_ids)

def _correlate_weighted(x1, x2, n_samples):
    """
    This is the method calculating the similarity score explained in the report.
    """
    t1 = np.zeros(n_samples)
    t2 = np.zeros(n_samples)
    t1[x1] = 1
    t2[x2] = 1
    corr = np.max(correlate(t1, t2, 'same'))
    return corr / np.sqrt(len(x1) * len(x2))

def _correlate_simple(x1, x2, n_samples, n):
    t1 = np.zeros(n_samples)
    t2 = np.zeros(n_samples)
    t1[x1] = 1
    t2[x2] = 1

    if n > 0:
        for i in range(1, n + 1):
            t2[x2 - i] = 1
            t2[x2 + i] = 1

    corr = correlate(t1, t2, 'same')
    return np.max(corr) / float(len(x1))

def _get_tpfn(actual, pred, n=0, samples=100000):
    """
    Manually counts TPs and FNs.
    """
    tp = 0
    fn = 0
    pred_t = np.zeros((samples))
    pred_t[pred] = 1
    for a in actual:
        start = max(0, a - n)
        end = min(samples, a + n + 1)
        if(np.any(pred_t[start:end])):
            tp += 1
        else:
            fn += 1
    return (tp, fn)

def _get_tpfp(actual, pred, n=0, samples=100000):
    """
    Manually counts TPs and FPs.
    """
    tp = 0
    fp = 0
    actual_t = np.zeros((samples))
    actual_t[actual] = 1
    for p in pred:
        start = max(0, p - n)
        end = min(samples, p + n + 1)
        if(np.any(actual_t[start:end])):
            tp += 1
        else:
            fp += 1
    return (tp, fp)