
import numpy as np
from pymatbridge import Matlab


def roundNearestQuarter(x):
    return 25*np.round(x/25)

def floorNearestQuarter(x):
    return 25*np.floor(x/25)


# def startMatlab():
#     # This function assumes MC-NMF is a subdirectory of working directory
#
#     mlab = Matlab()
#     mlab.start()
#
#     res = mlab.run_code("cd ../MC-NMF/; path(path,genpath(pwd)); cd ..")
#     if res['success']:
#         print("Matlab connection succesfully started")
#         return mlab
#     else:
#         print("Error starting Matlab")

from .matrix_completion import *
from .plotting import *
from .partial_least_squares import *
