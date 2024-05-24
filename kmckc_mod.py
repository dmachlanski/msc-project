"""
This is an experimental decomposition method that combines KmCKC [1] and the refinement loop from [2] based on CoV.
See kmckc.py for more details on how the KmCKC itself works.

[1] Y. Ning, X. Zhu, S. Zhu, and Y. Zhang, ‘Surface EMG Decomposition Based on K-means Clustering and Convolution Kernel Compensation’, IEEE Journal of Biomedical and Health Informatics, vol. 19, no. 2, pp. 471–477, Mar. 2015, doi: 10.1109/JBHI.2014.2328497.
[2] F. Negro, S. Muceli, A. M. Castronovo, A. Holobar, and D. Farina, ‘Multi-channel intramuscular and surface EMG decomposition by convolutive blind source separation’, J Neural Eng, vol. 13, no. 2, p. 026027, Apr. 2016, doi: 10.1088/1741-2560/13/2/026027.
"""

import numpy as np
import random
from sklearn.cluster import KMeans
from base import Base
from utils import find_highest_peaks, get_cov_isi, get_peaks_single

class KmCKC_mod(Base):

    def decompose_alt(self, options):
        return self.decompose(Nmdl=options.km_nmdl, r=options.km_r, Np=options.km_np,
                            h=options.km_h, ipt_th=options.km_ipt_th, rk=options.km_rk,
                            verbose=options.verbose)

    def decompose(self, Nmdl=150, r=10, Np=10, h=20, ipt_th=200, rk=300, c=2, verbose=0, peak_dist=1):
        super().decompose()
        results = []

        for i in range(Nmdl):
            if((verbose > 3) or (verbose > 2 and i%10 == 0)): print(f"KmCKC: {i}/{Nmdl}")
            
            n0 = np.argsort(self.activity_index)[len(self.activity_index)//2]
            s_n0 = self._reconstruct_signal(n0).flatten()

            # Find max value in s_n0 and its time of occurence
            n1 = np.argmax(s_n0)
            #n1 = random.choice(find_peaks(s_n0, height=3.0*np.std(s_n0))[0])

            s_n1 = self._reconstruct_signal(n1).flatten()
            p_nc = find_highest_peaks(s_n1, count=rk, height=0, dist=peak_dist)

            kmeans = KMeans(n_clusters=c).fit(self.x[:, p_nc].T)
            largest_group = np.bincount(kmeans.labels_).argmax()

            largest_group_local_ids = np.where(kmeans.labels_ == largest_group)[0]
            p_nv = p_nc[largest_group_local_ids]

            Cx = np.sum(self.x[:, p_nv], axis=1, keepdims=True)/len(p_nv)
            ipt = np.matmul(np.matmul(Cx.T, self.Cxx_inv), self.x).flatten()

            local_r = r
            for _ in range(h):
                p_nu = find_highest_peaks(ipt, count=local_r, height=0, dist=peak_dist)
                if len(p_nu) < 1:
                    break

                Cx = np.sum(self.x[:, p_nu], axis=1, keepdims=True)/len(p_nu)
                ipt = np.matmul(np.matmul(Cx.T, self.Cxx_inv), self.x).flatten()
                local_r += Np

            # This is where KmCKC ends and the other refinement phase begins.

            p_nu = get_peaks_single(ipt)
            cov_prev = 100.0
            cov_curr = get_cov_isi(p_nu)
            
            while(cov_curr < cov_prev):
                best_Cx = Cx
                best_ipt = ipt
                Cx = np.sum(self.x[:, p_nu], axis=1, keepdims=True)/len(p_nu)
                ipt = np.matmul(np.matmul(Cx.T, self.Cxx_inv), self.x).flatten()
                p_nu = get_peaks_single(ipt)
                cov_prev = cov_curr
                cov_curr = get_cov_isi(p_nu)

            results.append({'ipt': best_ipt, 'sep': best_Cx})

            #ai_n = 2
            self.activity_index[p_nv] = 0
            #for n in range(1, ai_n+1):
                #self.activity_index[p_nv - n] = 0
                #self.activity_index[p_nv + n] = 0

        return results
