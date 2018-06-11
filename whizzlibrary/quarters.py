
import numpy as np



def roundNearestQuarter(x):
    return 25*np.round(x/25)

def floorNearestQuarter(x):
    return 25*np.floor(x/25)


def histogramQuarters(x):
    sorted_array = np.sort(x)
    m, M = sorted_array[0], sorted_array[-1]
    bins = np.arange(m, M+26,  25) - 12.5

    counts = np.zeros(len(bins) - 1)
    i = 0

    for entry in sorted_array[:-1]:   # we treat last element as a special case
        if entry < bins[i+1]:
            counts[i] += 1
        else:                        # we do a search
            while entry > bins[i+1]:
                i += 1
            counts[i] += 1

    counts[-1] += 1                 # last element will always be in last box

    return counts, bins


def errorStatistics(completed_vec, original_vec, verbose=True):
    # completed_vec should already have entries that are quarters

    signed_errors = completed_vec - original_vec
    counts, bins = histogramQuarters(signed_errors)

    centers = bins[:-1] + 12.5
    zro = np.where(centers == 0.0)[0][0]

    if len(counts) >= 5:
        output = np.array([ counts[zro], np.sum(counts[zro-1:zro+2]), np.sum(counts[zro-2:zro+3]),
                            np.sum(counts[:zro]), np.sum(counts[zro+1:]) ])
    elif len(counts) >= 3:
        output = np.array([ counts[zro], np.sum(counts[zro-1:zro+2]), len(signed_errors),
                            np.sum(counts[:zro]), np.sum(counts[zro+1:]) ])
    else:
        output = np.array([ counts[zro], len(signed_errors), len(signed_errors),
                            np.sum(counts[:zro]), np.sum(counts[zro+1:]) ])
    output *= 100 / len(signed_errors)

    rmse = np.linalg.norm(signed_errors, 2) / np.sqrt(len(signed_errors))
    output = np.append(output, rmse)

    if verbose:
        print('Exact: \t\t %.2f %%' %  output[0])
        print('Within 25:\t %.2f %%' % output[1] )
        print('Within 50:\t %.2f %%' % output[2] )
        print('Underestimated:\t %.2f %%' % output[3])
        print('Overestimated:\t %.2f %%' % output[4])
        print('Min: \t\t %d' % centers[0])
        print('Max: \t\t %d' % centers[-1])
        print('RMSE: \t\t %.2f' % output[5])

    return output
