
import numpy as np
from .quarters import roundNearestQuarter, floorNearestQuarter, errorStatistics



def kTopicsOut(mat, k, seed=0):
    nb_topics, nb_pupils = mat.shape

    np.random.seed(seed)
    idx_i = np.array([np.random.choice(nb_topics, size=k, replace=False) for _ in range(nb_pupils)]).T.reshape(-1) # T because the choices for a student are along rows, and reshape concatenates cols
    idx_j = np.array(list(range(nb_pupils)) * k)

    incomplete_mat = mat.copy()
    incomplete_mat[idx_i, idx_j] = 0.0

    return incomplete_mat, idx_i, idx_j


def nonnegativeMatrixCompletion(connector, incomplete_mat, idx, rank_estimate, alg_tol=1e-8,
                                nearest_quarter='round'):
    # The zero entries
    idx_i, idx_j = idx

    # to select the observations (non-zero entries)
    mask = np.ones(incomplete_mat.shape, dtype=bool)
    mask[idx_i, idx_j] = 0

    res = connector.run_func('callMCNMF.m', {'mat': incomplete_mat, 'rank_estimate': rank_estimate, 'seed': 0, 'alg_tol': alg_tol}, nargout=2)
    filled_mat, iters = res['result']

    print('Iterations: ', iters)

    if nearest_quarter == 'floor':
        filled_mat = floorNearestQuarter(filled_mat) # whizz rounds down
    elif nearest_quarter == 'round' or nearest_quarter not in ['round', 'floor']:
        filled_mat = roundNearestQuarter(filled_mat)

    return filled_mat[idx_i, idx_j], filled_mat[mask]


def repeatMatrixCompletion(connector, mat, k, rank_estimate, alg_tol=1e-8, nb_repeats=1,
                           nearest_quarter='round', verbose=True):
    error_stats = np.zeros(6)

    for i in range(nb_repeats):
        incomplete_mat, idx_i, idx_j = kTopicsOut(mat, k, seed=i)

        # to select the observations (non-zero entries)
        mask = np.ones(mat.shape, dtype=bool)
        mask[idx_i, idx_j] = 0

        filled, recovered_obs = nonnegativeMatrixCompletion(connector,
        incomplete_mat, (idx_i, idx_j), rank_estimate, alg_tol=alg_tol, nearest_quarter=nearest_quarter)

        original, obs = mat[idx_i, idx_j], mat[mask]

        error_stats += errorStatistics(filled, original, verbose=False)
        print('Observations recovered: ', errorStatistics(recovered_obs, obs, verbose=False)[0])

    error_stats /= nb_repeats

    if verbose:
        print('Exact: \t\t %.2f %%' %  error_stats[0])
        print('Within 25:\t %.2f %%' % error_stats[1] )
        print('Within 50:\t %.2f %%' % error_stats[2] )
        print('Underestimated:\t %.2f %%' % error_stats[3])
        print('Overestimated:\t %.2f %%' % error_stats[4])
        print('RMSE: \t\t %.2f' % error_stats[5])


def meanFill(incomplete_mat, idx, nearest_quarter='round'):
    # The zero entries
    idx_i, idx_j = idx

    nan_mat = incomplete_mat.copy()
    nan_mat[idx_i, idx_j] = np.nan
    col_avg = np.nanmean(nan_mat, axis=0)

    for i in range(len(col_avg)):
        col = nan_mat[:,i]
        nan_mat[np.isnan(col), i] = col_avg[i]

    filled_entries = nan_mat[idx_i, idx_j]

    if nearest_quarter == 'floor':
        filled_entries = floorNearestQuarter(filled_entries) # whizz rounds down
    elif nearest_quarter == 'round' or nearest_quarter not in ['round', 'floor']:
        filled_entries = roundNearestQuarter(filled_entries)

    return filled_entries


def repeatMeanFill(mat, budget_k, nb_repeats=1, nearest_quarter='round', verbose=True):
    error_stats = np.zeros(6)

    for i in range(nb_repeats):
        incomplete_mat, idx_i, idx_j = kTopicsOut(mat, budget_k, seed=i)

        filled = meanFill(incomplete_mat, (idx_i, idx_j), nearest_quarter=nearest_quarter)
        original = mat[idx_i, idx_j]

        error_stats += errorStatistics(filled, original, verbose=False)

    error_stats /= nb_repeats

    if verbose:
        print('(%d repeats)'% nb_repeats)
        print('Exact: \t\t %.2f %%' %  error_stats[0])
        print('Within 25:\t %.2f %%' % error_stats[1] )
        print('Within 50:\t %.2f %%' % error_stats[2] )
        print('Underestimated:\t %.2f %%' % error_stats[3])
        print('Overestimated:\t %.2f %%' % error_stats[4])
        print('RMSE: \t\t %.2f' % error_stats[5])

    return error_stats
