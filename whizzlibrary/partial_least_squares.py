
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from . import roundNearestQuarter, floorNearestQuarter



def repeatPLS(range_mat, known_topics, topic_names, nb_repeats=1, nb_components=2,
              mean_fill=True, quarter_round=True, verbose=True):
    if mean_fill:
        filled_mat = range_mat.copy()
        filled_mat[filled_mat == 0] = np.nan
        # nan_mat = filled_mat.copy() # not used

        means = np.nanmean(filled_mat, axis=0)

        idx = np.argwhere(np.isnan(filled_mat))
        idx_i, idx_j = idx[:,0], idx[:,1]
        filled_mat[idx_i, idx_j] = floorNearestQuarter(means[idx_j]) # rounding here makes no significant difference
    else:
        filled_mat = range_mat.copy()

    nb_topics, _ = range_mat.shape

    mask = np.zeros(nb_topics, dtype=bool)
    mask[known_topics] = 1
    k = np.sum(~mask)                       # nb of removed topics

    X = filled_mat[mask, :].T               # tranpose to use each row as observation in sklearn
    Y = filled_mat[~mask, :].T

    # the average of only the known topics
    # X_average = np.nanmean(nan_mat[mask, :], axis=0)
    # I don't use this because I need to split it with X and Y and this is
    # much more complicated. Instead I use direactly the mean of X_test

    rmse = np.zeros(k)
    percentage_wrong = np.zeros(k)
    comparison_mean = np.zeros(k)

    pls = PLSRegression(n_components=nb_components)

    for i in range(nb_repeats):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42 + i)
        X_average = np.mean(X_test, axis=1)   # along columns because X_test is transposed

        pls.fit(X_train, Y_train)
        prediction = pls.predict(X_test)

        if quarter_round:
            # prediction = floorNearestQuarter(prediction) # whizz rounds down, but this seems to be much worse in our case
            prediction = roundNearestQuarter(prediction)

        for t in range(k):
            err_t = Y_test[:,t] - prediction[:,t]
            rmse[t] += np.linalg.norm(err_t, 2) / np.sqrt(len(prediction))

            percentage_wrong[t] += np.linalg.norm(err_t, 0) *100 / len(prediction)

            err_mean = Y_test[:,t] - X_average
            comparison_mean[t] += np.linalg.norm(err_mean, 2) / np.sqrt(len(prediction))

    rmse /= nb_repeats
    percentage_wrong /= nb_repeats
    comparison_mean /= nb_repeats

    if verbose:
        print('Known topics: %s (%d repeats)' % (' - '.join( topic_names[known_topics].tolist() ), nb_repeats) )
        for t, name in enumerate(topic_names[~mask]):
            print('%s: \t %.1f \t %.1f \t %.1f' %(name, rmse[t], percentage_wrong[t], comparison_mean[t]) )
        print('Avg: \t %.1f \t %.1f \t %.1f' %(np.mean(rmse), np.mean(percentage_wrong), np.mean(comparison_mean)) )

    return (rmse, percentage_wrong, comparison_mean)


def testCombinations(range_mat, k, topic_names, nb_repeats=1, nb_components=2,
                     mean_fill=True, quarter_round=True, verbose=True):
    nb_topics, _ = range_mat.shape
    topic_combs = itertools.combinations(range(nb_topics), nb_topics - k)
    topic_combs = np.array(list(topic_combs))                 # coverted to array to use as indexes of topic names

    errors = np.zeros(len(topic_combs))

    for i, known_topics in enumerate(topic_combs):

        rmse, _, _ = repeatPLS(range_mat, known_topics, topic_names, nb_repeats=nb_repeats,
                               nb_components=nb_components, mean_fill=mean_fill,
                               quarter_round=quarter_round, verbose=verbose)
        errors[i] = np.mean(rmse)

    best = np.argmin(errors)
    return (errors[best], topic_names[topic_combs[best]])
