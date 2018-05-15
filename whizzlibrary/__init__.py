
import numpy as np
from pymatbridge import Matlab

__all__ = ["startMatlab", "roundNearestQuarter", "floorNearestQuarter",
           "errorStatistics", "kTopicsOut", "repeatMatrixCompletion", "histogramQuarters",
           "PLSkTopicsOut", "repeatPLS", "testCombinations",
           "plotSingInfo"]


def roundNearestQuarter(x):
    return 25*np.round(x/25)

def floorNearestQuarter(x):
    return 25*np.floor(x/25)


def startMatlab(nonnegative_dir):
    mlab = Matlab()
    mlab.start()

    res = mlab.run_code("path(path,genpath('%s'))" % nonnegative_dir)
    if res['success']:
        print("Matlab connection succesfully started")
        return mlab
    else:
        print("Error starting Matlab")

from .matrix_completion import *
from .plotting import *
from .partial_least_squares import *
