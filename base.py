"""
This is a base class that is implemented/inherited by actual implementations of decomposition algorithms (CKC, KmCKC).

It includes common code shared among all the algorithms to reduce code duplication. This includes:
- Data extension.
- Data whitening.
- Calculation of the auto-correlation matrix (Cxx and Cxx_inv).
- Calculation of the activity index.
"""

import numpy as np
from abc import ABC, abstractmethod
from utils import extend_data, whiten_data

class Base(ABC):
    
    def __init__(self, x, delays=0, step=1, whiten=True):
        # Extend the observations.
        if(delays > 0):
            x_ex = extend_data(x, delays, step)
        else:
            x_ex = x
        
        # Perform whitening.
        if whiten:
            self.x, noise_var = whiten_data(x_ex)
        else:
            self.x = x_ex
            noise_var = 0

        # Calculate the auto-correlation matrix.
        Cxx = np.matmul(self.x, self.x.T)/float(self.x.shape[1])

        # And find its inverse (a matrix can be non-inversible).
        try:
            self.Cxx_inv = np.linalg.inv(Cxx)
        except np.linalg.LinAlgError:
            print("Cannot invert correlation matrix Cxx")
            return
        
        # Noise threshold computation used specifically by CKC only.
        self.noise_thr = noise_var * np.linalg.norm(self.Cxx_inv, ord=1)
        
    def _reconstruct_signal(self, index):
        """
        Vector reconstruction as given by Equation (9) in the CKC paper.
        """
        x_i_T = (self.x[:, index, None]).T
        return np.matmul(np.matmul(x_i_T, self.Cxx_inv), self.x)

    @abstractmethod
    def decompose(self):
        'Decomposes input signals into approximate sources'
        
        # Compute activity index
        samples_count = self.x.shape[1]
        self.activity_index = np.empty(samples_count)

        # Slow but memory-efficient
        for i in range(samples_count):
            x_n = self.x[:, i, None]
            self.activity_index[i] = np.matmul(np.matmul(x_n.T, self.Cxx_inv), x_n)

        # Alternative (probably faster but less memory-efficient)
        #activity_index = np.diag(np.matmul(np.matmul(x.T, Cxx_inv), x))