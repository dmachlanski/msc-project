import numpy as np
from base import Base
from utils import get_cov_isi, get_peaks_single

class negro(Base):

    def decompose_alt(self, options):
        return self.decompose(options.itermax, verbose=options.verbose)

    def decompose(self, itermax, verbose=0):
        super().decompose()
        results = []

        for i in range(itermax):
            if((verbose > 3) or (verbose > 2 and i%10 == 0)): print(f"fastICA: {i}/{itermax}")
            
            n0 = np.argsort(self.activity_index)[len(self.activity_index)//2]
            s_n0 = self._reconstruct_signal(n0).flatten()

            # Find max value in s_n0 and its time of occurence
            n1 = np.argmax(s_n0)
            
            w0 = self.x[:, n1]

            #--- main code---

            

            # end of main code

            #--- refinement step---

            #Cx = np.sum(self.x[:, p_nv], axis=1, keepdims=True)/len(p_nv)
            ipt = np.matmul(np.matmul(Cx.T, self.Cxx_inv), self.x).flatten()

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

            self.activity_index[p_nu] = 0

        return results
