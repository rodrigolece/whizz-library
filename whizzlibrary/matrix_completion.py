
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["errorStatistics", "kTopicsOut", "repeatMatrixCompletion"]



def errorStatistics(completed_vec, original_vec, verbose=True):
    # completed_vec should already have entries that are quarters

    signed_errors = completed_vec - original_vec

    m, M = min(signed_errors), max(signed_errors)
    centers = np.arange(m, M+26,  25)            # one exra point to use centers as a basis for the bins

    counts, _, _ = plt.hist(signed_errors, bins=(centers-12.5))

    zro = np.where(centers == 0)[0][0]

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
        print('Min: \t\t %d' % m)
        print('Max: \t\t %d' % M)
        print('RMSE: \t\t %.2f' % output[5])

    return output


def kTopicsOut(mat, k, seed=0):
    nb_topics, nb_pupils = mat.shape

    np.random.seed(seed)
    idx_i = np.array([np.random.choice(nb_topics, size=k, replace=False) for _ in range(nb_pupils)]).T.reshape(-1) # transpose because the entries of a student are along rows
    idx_j = np.array(list(range(nb_pupils)) * k)

    incomplete_mat = mat.copy()
    incomplete_mat[idx_i, idx_j] = 0.0

    return incomplete_mat, idx_i, idx_j


def repeatMatrixCompletion(connector, mat, k, rank_estimate, alg_tol = 1e-8, nb_repeats=1, quarter_round=True, verbose=True):

    average_percentages = np.zeros(6)

    for i in range(nb_repeats):
        incomplete_mat, idx_i, idx_j = kTopicsOut(mat, k, seed=i)

        res = connector.run_func('callMCNMF.m', {'range_mat': incomplete_mat, 'rank_estimate': rank_estimate, 'seed': 0, 'alg_tol': alg_tol}, nargout=2)
        filled_mat, iters = res['result']

        # print('Iter: ', i, '\t Algorithm returned after: ', iters)

        if iters < 500:
            print("Low iters, possible non-convergence")

        if quarter_round:
            filled_mat = roundNearestQuarter(filled_mat)

        filled_vec = filled_mat[idx_i, idx_j]
        original_vec = mat[idx_i, idx_j]

        average_percentages += errorStatistics(filled_vec, original_vec, verbose=False)

    average_percentages /= nb_repeats

    if verbose:
        print('Exact: \t\t %.2f %%' %  average_percentages[0])
        print('Within 25:\t %.2f %%' % average_percentages[1] )
        print('Within 50:\t %.2f %%' % average_percentages[2] )
        print('Underestimated:\t %.2f %%' % average_percentages[3])
        print('Overestimated:\t %.2f %%' % average_percentages[4])
        print('RMSE: \t\t %.2f' % average_percentages[5])
