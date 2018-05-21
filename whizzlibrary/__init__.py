
from pymatbridge import Matlab

__all__ = ["startMatlab", "roundNearestQuarter", "floorNearestQuarter",
           "histogramQuarters", "errorStatistics", "kTopicsOut",
           "nonnegativeMatrixCompletion", "repeatMatrixCompletion",
           "PLSkTopicsOut", "repeatPLS", "testCombinations",
           "plotTopicHistograms", "plotSingInfo", "correlationMat", "plotCorrelations"]




def startMatlab(nonnegative_dir, hardthresh_dir):
    mlab = Matlab()
    mlab.start()

    # I could do a cd here to make sure that the call functions are in the working dir

    status1 = mlab.run_code("addpath %s" % nonnegative_dir)['success']
    status2 = mlab.run_code("addpath %s/Main" % hardthresh_dir)
    status3 = mlab.run_code("addpath %s/Auxiliary" % hardthresh_dir)

    if status1 and status2 and status3:
        print("Libraries succesfully loaded")
    else:
        print("Error loading libraries")
        return

    return mlab

from .quarters import *
from .matrix_completion import *
from .plotting import *
from .partial_least_squares import *
