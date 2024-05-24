import numpy as np
import os
import matlab
import matlab.engine

def isctest(components, exec_path, alpha=0.001):
    """
    Calls the ISCTEST method via matlab engine as the method is written in matlab.
    Might be useful to translate it into python someday to eliminate the matlab dependency.
    """
    path = os.getcwd()
    os.chdir(exec_path)
    eng = matlab.engine.start_matlab()
    spatialPattTens = matlab.double(components.tolist())
    clustering = eng.isctest(spatialPattTens, alpha, alpha, 'components', 'silent')
    eng.quit()
    os.chdir(path)
    return np.asarray(clustering, dtype=int)