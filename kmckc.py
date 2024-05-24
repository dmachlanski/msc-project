"""
This class implements the KmCKC decomposition algorithm introduced in:
Y. Ning, X. Zhu, S. Zhu, and Y. Zhang, ‘Surface EMG Decomposition Based on K-means Clustering and Convolution Kernel Compensation’, IEEE Journal of Biomedical and Health Informatics, vol. 19, no. 2, pp. 471–477, Mar. 2015, doi: 10.1109/JBHI.2014.2328497.
"""

import numpy as np
import random
from sklearn.cluster import KMeans
from base import Base
from utils import find_highest_peaks

class KmCKC(Base):

    def decompose_alt(self, options):
        return self.decompose(Nmdl=options.km_nmdl, r=options.km_r, Np=options.km_np,
                            h=options.km_h, rk=options.km_rk, verbose=options.verbose)

    def decompose(self, Nmdl=150, r=10, Np=10, h=20, rk=300, c=2, verbose=0, peak_dist=1):
        super().decompose()
        results = []

        for i in range(Nmdl):
            if((verbose > 3) or (verbose > 2 and i%10 == 0)): print(f"KmCKC: {i}/{Nmdl}")
            
            # Take the time instant corresponding to the median value of the activity index.
            # This is the starting point.
            n0 = np.argsort(self.activity_index)[len(self.activity_index)//2]

            # Reconstruct the vector for the starting point.
            s_n0 = self._reconstruct_signal(n0).flatten()

            # Find max value in s_n0 and its time of occurence.
            # Classic CKC takes a random value here.
            n1 = np.argmax(s_n0)

            # Reconstruct the vector again and find 'rk' highest peaks.
            # In practice rk=60 works usually the best.
            s_n1 = self._reconstruct_signal(n1).flatten()
            p_nc = find_highest_peaks(s_n1, count=rk, height=0, dist=peak_dist)

            # The K-means clustering part.
            kmeans = KMeans(n_clusters=c).fit(self.x[:, p_nc].T)
            largest_group = np.bincount(kmeans.labels_).argmax()
            largest_group_local_ids = np.where(kmeans.labels_ == largest_group)[0]
            p_nv = p_nc[largest_group_local_ids]

            # Use the time instants of the selected group to reconstruct a full (but initial) IPT.
            Cx_sj0 = np.sum(self.x[:, p_nv], axis=1, keepdims=True)/len(p_nv)

            # Estimate entire j-th pulse train
            s_jh = np.matmul(np.matmul(Cx_sj0.T, self.Cxx_inv), self.x).flatten()

            # The refinement phase.
            local_r = r
            for _ in range(h):
                p_nu = find_highest_peaks(s_jh, count=local_r, height=0, dist=peak_dist)
                if len(p_nu) < 1:
                    break

                Cx_sjh = np.sum(self.x[:, p_nu], axis=1, keepdims=True)/len(p_nu)
                s_jh = np.matmul(np.matmul(Cx_sjh.T, self.Cxx_inv), self.x).flatten()
                local_r += Np

            # Save the resulting IPT and its corresponding separation vector.
            results.append({'ipt': s_jh, 'sep': Cx_sjh})

            # Update the activity index.
            self.activity_index[p_nv] = 0

        return results
