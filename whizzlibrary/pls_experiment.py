
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from .topic_sequencer import TopicSequencer
from .quarters import errorStatistics, roundNearestQuarter, floorNearestQuarter



def idxTrainTestSplit(sample_size, seed=0):
    np.random.seed(seed)

    idx_test = np.sort(np.random.choice(range(sample_size), int(sample_size/3)))

    mask = np.ones(sample_size,dtype=bool)
    mask[idx_test] = 0
    idx_train = np.where(mask)[0]

    return (idx_train, idx_test)

def assessStudents(mat, range_topics, budget_k, verbose=False):
    _, nb_pupils = mat.shape
    nb_filled = len(range_topics) - budget_k

    sequencers = np.array([TopicSequencer(budget_k, mat[:,i]) for i in range(nb_pupils)])

    sequencer_class = np.zeros(nb_pupils, dtype=int)
    incompatibilities = np.zeros(nb_pupils, dtype=bool)

    for i, seq in enumerate(sequencers):
        seq.initialise()
        seq.assessAll()

        sequencer_class[i] = seq.finalRange()
        incompatibilities[i] = seq._TopicSequencer__incompatibility
    if verbose:
        print('Nb of incompatibilities encountered:', np.sum(incompatibilities))

    # We dicard incompatible data
    return sequencers[~incompatibilities], sequencer_class[~incompatibilities]

def getModels(sequencers, range_topics, budget_k):
    nb_pupils = len(sequencers)
    model_topics = np.zeros((budget_k, nb_pupils), dtype=int)

    for i, seq in enumerate(sequencers):
        # This would normally break down if the assesment is not completed and
        # the budget is not reached, but the incompatible data is not considered anymore
        model_topics[:,i] = np.sort(seq.assessed)

    models, model_idx = np.unique(model_topics, axis=1, return_inverse=True)

    return models, model_idx

def getMatrices(sequencers, range_topics, budget_k):
    nb_pupils = len(sequencers)
    nb_filled = len(range_topics) - budget_k

    X = np.zeros((budget_k, nb_pupils))
    Y = np.zeros((nb_filled, nb_pupils))


    for i, seq in enumerate(sequencers):
        X[:,i] = seq.student_profile[np.sort(seq.assessed)]
        if len(seq.unassessed) != nb_filled:
            target = set(range_topics).difference(set(seq.assessed))
            if target.issubset(set(seq.unassessed)):
                Y[:,i] = seq.student_profile[list(target)]
        else:
            Y[:,i] = seq.student_profile[np.sort(seq.unassessed)]

    return X, Y

class PLSmodel(object):
    def __init__(self, mat, range_topics, known_topics, nb_components=2):
        self.known_topics = known_topics
        predict_topics = sorted(set(range_topics).difference(set(known_topics)))
        self.predict_topics = predict_topics

        X = mat[known_topics, :].T
        Y = mat[predict_topics,:].T

        model = PLSRegression(n_components=nb_components)
        model.fit(X,Y)
        self.__model = model


    def predict(self, X):
        prediction = self.__model.predict(X.T)
        return roundNearestQuarter(prediction).T


def plsModels(mat, range_topics, models, nb_components=2):
    budget_k, nb_models = models.shape
    nb_filled = len(range_topics) - budget_k

    trained_models = []

    for i in range(nb_models):
        known_topics = models[:,i]

        model = PLSmodel(mat, range_topics, known_topics, nb_components=nb_components)
        trained_models.append(model)

    return trained_models


def rangeExperimentPLS(range_mat, range_topics, budget_k, seed=0, nb_components=2, verbose=True):
    nb_topics, sample_size  = range_mat.shape

    assert nb_topics == 13

    if budget_k == 1:
        nb_components = 1

    train, test = idxTrainTestSplit(sample_size, seed=seed)
    mat_train = range_mat[:,train]
    mat_test = range_mat[:,test]

    sequencers, seq_class = assessStudents(mat_test, range_topics, budget_k, verbose=verbose)

    models, model_idx = getModels(sequencers, range_topics, budget_k)

    X, Y = getMatrices(sequencers, range_topics, budget_k)

    # We discard missing data
    idx = np.all(Y > 0, axis=0)
    X = X[:,idx]
    Y = Y[:,idx]
    model_idx = model_idx[idx]

    counts_each_model = [np.sum(model_idx == i) for i in range(models.shape[1])]

    pls_models = plsModels(mat_train, range_topics, models, nb_components=nb_components)
    nb_models = len(pls_models)

    original_mat = np.zeros(Y.shape)
    filled_mat = np.zeros(Y.shape)

    start = 0

    for i in range(nb_models):
        if counts_each_model[i] == 0:
            continue
        idx = (model_idx == i)
        original_entries = Y[:,idx]
        filled_entries = pls_models[i].predict(X[:,idx])

        end = np.cumsum(counts_each_model)[i]
        original_mat[:,start:end] = original_entries
        filled_mat[:, start:end] = filled_entries
        start = end

    # as the current range we take the most common
    current_range = max(np.unique(seq_class), key=list(seq_class).count) # key modifies the rule for comparison by applying a function to each element

    if verbose:
        print('Correctly classified: %.1f %%' %(np.sum(seq_class == current_range)*100/len(test)))
        print('Counts in each model:', counts_each_model)

    return filled_mat, original_mat, np.sort(model_idx)


def repeatRangeExperimentPLS(range_mat, range_topics, budget_k, nb_repeats=1, nb_components=2, verbose=True):
    nb_topics, _  = range_mat.shape

    assert nb_topics == 13

    error_stats = np.zeros(6)

    for i in range(nb_repeats):
        # third argument can be used to break up by model
        filled_mat, original_mat, _ = rangeExperimentPLS(range_mat, range_topics, budget_k, seed=i, verbose=False)
        error_stats += errorStatistics(filled_mat.reshape(-1), original_mat.reshape(-1), verbose=False)

    error_stats /= nb_repeats

    if verbose:
        print('(%d repeats)' % nb_repeats)

        print('Exact: \t\t %.2f %%' %  error_stats[0])
        print('Within 25:\t %.2f %%' % error_stats[1] )
        print('Within 50:\t %.2f %%' % error_stats[2] )
        print('Underestimated:\t %.2f %%' % error_stats[3])
        print('Overestimated:\t %.2f %%' % error_stats[4])
        print('RMSE: \t\t %.2f' % error_stats[5])

    return error_stats
