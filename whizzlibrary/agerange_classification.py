
import numpy as np



def runningMathsAge(mat, final=False):
    rma = np.cumsum(mat, axis=0) / np.cumsum(mat > 0, axis=0)

    if final:
        return rma[-1]    # for an array this is last row
    else:
        return rma


def classifyInQuarters(x):
    # TODO: change this name
    if 500 <= x and x < 700:
        return 0
    elif 700 <= x and x < 800:
        return 1
    elif 800 <= x and x < 900:
        return 2
    elif 900 <= x and x < 1000:
        return 3
    elif 1000 <= x and x < 1200:
        return 4
    elif 1200 <= x and x <= 1400:  # there is one entry equal to 1400
        return 5


def compareMatching(a, b, nb_ranges=6):
    matching = a[a == b]

    return np.array([np.sum(matching == i) for i in range(nb_ranges)])


def correctPredictions(mat, age_range, topics, weight, target_class):
    m, M = age_range
    first_topic, second_topic = topics

    idx = np.where(np.bitwise_and(mat[first_topic,:] >= m, mat[first_topic,:] < M))[0]
    idx_nonzero = idx[np.nonzero(mat[second_topic,:][idx])]
    subset_mat = mat[:,idx_nonzero]

    avg = np.average(subset_mat[[first_topic, second_topic], :], weights=[weight,1-weight], axis=0)
    classification = np.fromiter(map(classifyInQuarters, avg), int)

    return compareMatching(classification, target_class[idx_nonzero])


def findSecondTopic(mat, age_range, target_class, first_topic=2, weight_step=0.1):
    nb_topics, _ = mat.shape
    topics = list(range(first_topic)) + list(range(first_topic+1, nb_topics))

    weights = np.arange(0,1,weight_step)

    combinations = np.zeros((len(topics), len(weights)))

    for i, second_topic in enumerate(topics):
        for j, w in enumerate(weights):
            correct_predictions = correctPredictions(mat, age_range, (first_topic, second_topic), w, target_class)
            combinations[i,j] = np.sum(correct_predictions)

    idx_i, idx_j = np.unravel_index(np.argmax(combinations), combinations.shape)
    best_topic, best_weight = topics[idx_i], weights[idx_j]

    return best_topic, best_weight


def classifyStudent(student):
    assert len(student) == 13 # this classifcation should be done with respect to the full mat

    first_topic = 2           # QA
    first_topic_class = classifyInQuarters(student[first_topic])

    if first_topic_class == 0:
        second_topic = 1
        w = 0.55
    elif first_topic_class == 1:
        second_topic = 0
        w = 0.6
    elif first_topic_class == 2:
        second_topic = 0
        w = 0.55
    elif first_topic_class == 3:
        second_topic = 4
        w = 0.75
    elif first_topic_class == 4:
        second_topic = 9
        w = 0.6
    elif first_topic_class == 5:
        second_topic = 10
        w = 0.55

    if student[second_topic] == 0:
        avg = student[first_topic]
    else:
        avg = np.average(student[[first_topic,second_topic]], weights=[w, 1-w])

    return classifyInQuarters(avg)


def applyModel(student, models, student_class=None):
    if student_class == None:
        student_class = classifyStudent(student)

    if student_class == 0:
        range_topics = [0,1,2,3]
        known_topics = [1,2]        # BA, QA
    elif student_class == 1:
        range_topics = [0,1,2,4,5]
        known_topics = [0,1,2]      # AA, BA, QA
    elif student_class == 2:
        range_topics = [0,1,2,4,6,7]
        known_topics = [0,1,2]      # AA, BA, QA
    elif student_class == 3:
        range_topics = [0,1,2,4,6,8,9]
        known_topics = [0,2,3,4]    # AA, QA, CA, GA
    elif student_class == 4:
        range_topics = [0,1,2,4,6,8,9,10]
        known_topics = [2,3,4,6]    # QA, CA, GA, LA
    elif student_class == 5:
        range_topics = [0,2,4,6,9,10,11,12]
        known_topics = [1,3,4,5]    # QA, GA, LA, SA

    student_classified = student[range_topics]

    nb_topics = len(range_topics)
    mask = np.zeros(nb_topics, dtype=bool)
    mask[known_topics] = 1

    x = student_classified[mask].reshape(1,-1) # this gives a row vector
    y_original = student_classified[~mask].reshape(1,-1)

    model = models[student_class]
    y = model.predict(x)

    return y, y_original
