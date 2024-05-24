"""
This class implements the CKC decomposition algorithm introduced in:
A. Holobar and D. Zazula, ‘Multichannel Blind Source Separation Using Convolution Kernel Compensation’, IEEE Transactions on Signal Processing, vol. 55, no. 9, pp. 4487–4496, Sep. 2007, doi: 10.1109/TSP.2007.896108.
"""

import numpy as np
import random
from base import Base
from scipy.signal import find_peaks
from itertools import combinations
from utils import find_highest_peaks

class CKC(Base):

    def decompose_alt(self, options):
        return self.decompose(J=options.const_j, itermax=options.itermax, verbose=options.verbose)

    def decompose(self, J, itermax=150, verbose=0):
        super().decompose()
        results = []
        
        adjusted_thr = np.mean(self.activity_index) + self.noise_thr
        self.activity_index[self.activity_index < adjusted_thr] = 0

        for i in range(itermax):
            if((verbose > 3) or (verbose > 2 and i%10 == 0)): print(f"CKC: {i}/{itermax}")

            n0 = np.argsort(self.activity_index)[len(self.activity_index)//2]

            # Calculate v_n0
            v_n0 = self._reconstruct_signal(n0).flatten()

            # Randomly choose one of its pulses (n1) exceeding the noise threshold
            n1 = random.choice(find_peaks(v_n0, height=self.noise_thr)[0])

            # Calculate v_n1
            v_n1 = self._reconstruct_signal(n1).flatten()

            h_n01 = np.multiply(v_n0, v_n1)

            # Collect all nr for which v_n0(nr) * v_n1(nr) > noise_threshold
            # len(nrs) == 7k
            # Take 200-300 highest peaks (or randomly)
            #nrs, _ = find_peaks(h_n01, noise_th)

            # The CKC instructs to take all the values exceeding the threshold. But this in practice gives 24 million iterations in a later stage (basically impractically slow given today's computation capabilities).
            nrs = find_highest_peaks(h_n01, count=200, height=self.noise_thr, dist=1)

            if verbose > 3: print('Reconstructing Nrs...')
            nrs_sig = {}
            for nr in nrs:
                nrs_sig[nr] = self._reconstruct_signal(nr).flatten()

            if verbose > 3: print('Trying all combinations...')
            candidates = set()
            # Binomial(7000, 2) == 24M !!!
            for n2, n3 in combinations(nrs, 2):
                # Avoid repeating if n2 and n3 are already candidates
                if n2 in candidates and n3 in candidates: continue

                h_nr = np.multiply(np.multiply(h_n01, nrs_sig[n2]), nrs_sig[n3])

                # Observe/detect the number of pulses for all combinations of nrs. Combinations of length 4 (n0, n1, nr_1, nr_2).
                # Any combination having more pulses than J is added to a set
                if len(find_peaks(h_nr, height=3.0*np.std(h_nr))[0]) > J:
                    candidates.add(n2)
                    candidates.add(n3)

            candidates_list = list(candidates)
            # Estimate entire j-th pulse train (eqs. 16 and 17).
            Cx_sj = np.sum(self.x[:, candidates_list], axis=1, keepdims=True)/len(candidates)
            s_j = np.matmul(np.matmul(Cx_sj.T, self.Cxx_inv), self.x).flatten()

            # Save the resulting IPT and its corresponding separation vector.
            results.append({'ipt': s_j, 'sep': Cx_sj})

            # Update the activity index to avoid converging to the same source again (well, it's not that simple but this is the main motivation).
            self.activity_index[candidates_list] = 0
            
        return results