
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

from .quarters import roundNearestQuarter, floorNearestQuarter, errorStatistics



def PLSkTopicsOut(mat, known_topics, seed=0):
    nb_topics, _ = mat.shape

    mask = np.zeros(nb_topics, dtype=bool)
    mask[known_topics] = 1

    X = mat[mask, :].T               # tranpose to use each row as observation in sklearn
    Y = mat[~mask, :].T

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

    return X_train, X_test, Y_train, Y_test


def repeatPLS(mat, known_topics, topic_names, nb_repeats=1, nb_components=2,
              nearest_quarter='round', split_topics=False, verbose=True):

    nb_topics, _ = mat.shape
    k = nb_topics - len(known_topics)                 # nb of removed topics

    topic_error_stats = np.zeros((6, k))

    pls = PLSRegression(n_components=nb_components)

    for i in range(nb_repeats):
        X_train, X_test, Y_train, Y_test = PLSkTopicsOut(mat, known_topics, seed=i)

        pls.fit(X_train, Y_train)
        filled_mat = pls.predict(X_test)

        if nearest_quarter == 'floor':
            filled_mat = floorNearestQuarter(filled_mat) # whizz rounds down
        elif nearest_quarter == 'round' or nearest_quarter not in ['round', 'floor']:
            filled_mat = roundNearestQuarter(filled_mat)

        for t in range(k):
            filled_vec = filled_mat[:,t]
            original_vec = Y_test[:,t]
            topic_error_stats[:,t] += errorStatistics(filled_vec, original_vec, verbose=False)

    topic_error_stats /= nb_repeats                  # this could be returned to do inspection by topic
    error_stats = np.mean(topic_error_stats, axis=1)

    if verbose:
        print('Known topics: %s (%d repeats)' % (' - '.join( topic_names[known_topics].tolist() ), nb_repeats) )

        print('Exact: \t\t %.2f %%' %  error_stats[0])
        print('Within 25:\t %.2f %%' % error_stats[1] )
        print('Within 50:\t %.2f %%' % error_stats[2] )
        print('Underestimated:\t %.2f %%' % error_stats[3])
        print('Overestimated:\t %.2f %%' % error_stats[4])
        print('RMSE: \t\t %.2f' % error_stats[5])

        # for t, name in enumerate(topic_names[~mask]):
        #     print('%s: \t %.1f \t %.1f \t %.1f' %(name, rmse[t], percentage_wrong[t], comparison_mean[t]) )
        # print('Avg: \t %.1f \t %.1f \t %.1f' %(np.mean(rmse), np.mean(percentage_wrong), np.mean(comparison_mean)) )

    if split_topics:
        return topic_error_stats
    
    avg_rmse = error_stats[5]
    return avg_rmse


def testCombinations(mat, k, topic_names, nb_repeats=1, nb_components=2,
                     nearest_quarter='round', verbose=True):

    nb_topics, _ = mat.shape
    topic_combs = itertools.combinations(range(nb_topics), nb_topics - k)
    topic_combs = np.array(list(topic_combs))                 # coverted to array to use as indexes of topic names

    avg_rmses = np.zeros(len(topic_combs))

    for i, known_topics in enumerate(topic_combs):
        avg_rmses[i] = repeatPLS(mat, known_topics, topic_names, nb_repeats=nb_repeats, nb_components=nb_components,
                                 nearest_quarter='round', verbose=verbose)

    best = np.argmin(avg_rmses)
    return (avg_rmses[best], topic_names[topic_combs[best]], topic_combs[best])
