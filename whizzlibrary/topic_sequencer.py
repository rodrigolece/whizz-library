
from .agerange_classification import runningMathsAge, classifyInQuarters



class TopicSequencer(object):
    def __init__(self, budget_k, student_profile):
        assert budget_k <= 8                          # currently no more than 8 topics are assessed

        self.budget_k = budget_k
        self.student_profile = student_profile        # this would later be changed by a dynamic acquisition of the ages
        self.nb_assessed = 0
        self.assessed = []


    def initialise(self):
        initial_sequences = self.__initialSequences()
        second_topics = self.__secondTopics()

        self.__range_sequences = tuple([self.__reorderSequence(seq, second_topics[i]) for i, seq in enumerate(initial_sequences)])

        self.__current_range = 0
        self.__current_sequence = self.__range_sequences[self.__current_range]

        self.__incompatibility = False
        self.__assessment_ended = False

        # return self.current_range, self.current_sequence


    def assessNext(self):
        topics_left = self.nb_assessed < self.budget_k and self.nb_assessed < len(self.__current_sequence)
        if topics_left and not self.__assessment_ended:
            current_set = set(self.assessed)
            new_set = set(self.__current_sequence[:self.nb_assessed+1])
            topic = new_set.difference(current_set).pop()

            # I have to do this because there might be missing data
            age = self.student_profile[topic]
            if age == 0:
                self.__incompatibility = True
                self.endAssessment()
                return

            self.assessed.append(topic)
            self.nb_assessed += 1

            rma = runningMathsAge(self.student_profile[self.assessed], final=True)
            new_range = classifyInQuarters(rma)

            if new_range != self.__current_range:
                new_sequence = self.__range_sequences[new_range]

                # we test compatibility of sequences
                if set(self.assessed).issubset(set(new_sequence)):
                    self.__current_range = new_range
                    self.__current_sequence = new_sequence
                else:
                    self.__incompatibility = True
                    self.endAssessment()
                    return

            # return self.current_range, self.current_sequence
            # return topic, self.student_profile[topic], self.__current_range

        else:
            self.endAssessment()
            return


    def endAssessment(self):
        # print('Ending the assessment')
        self.__assessment_ended = True
        self.unassessed = list(set(self.__current_sequence).difference(set(self.assessed)))


    def assessAll(self):
        while self.__assessment_ended == False:
            self.assessNext()
        return self.assessed

    def finalRange(self):
        if self.__assessment_ended:
            return self.__current_range
        else:
            self.assessAll()
            return self.__current_range


    def __initialSequences(self):
        seq57 = [0,3,1,2]
        seq78 = [1,0,2,4,5]
        seq89 = [0,1,4,6,2,7]
        seq910 = [6,0,4,2,1,8,9]
        seq1012 = [9,4,6,0,10,2,8,1]
        seq1214 = [6,9,0,10,11,4,2,12]

        out = (seq57, seq78, seq89, seq910, seq1012, seq1214)

        return out


    def __secondTopics(self):
        out = (1, 0, 0, 4, 9, 10)     # none should be equal to 2 (QA)

        return out


    def __reorderSequence(self, topic_sequence, second, first=2, inplace=False):   # first is by default QA
        # indices should be with respecto to 13 topics

        assert second != first

        if inplace:
            out = topic_sequence
        else:
            out = topic_sequence.copy()

        fidx = out.index(first)
        out.pop(fidx)
        out.insert(0, first)

        sidx = out.index(second)
        out.pop(sidx)
        out.insert(1, second)

        return out


    def __removeFromSequence(self, topic_sequence, topic, inplace=False):
        if inplace:
            out = topic_sequence
        else:
            out = topic_sequence.copy()

        idx = out.index(topic)
        out.pop(idx)

        return out
