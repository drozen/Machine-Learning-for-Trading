"""
BagLearner  by Daniel Rozen
"""
import math
import numpy as np
import KNNLearner as knn

class BagLearner(object):

    def __init__(self, learner = knn.KNNLearner, kwargs = {}, bags = 20, boost = False):
        """
        :param learner: learner is the learning class to use with bagging.
        :param kwargs: kwargs are keyword arguments to be passed on to the learner's constructor and they vary according to the learner
        :param bags: "bags" is the number of learners you should train using Bootstrap Aggregation
        :param boost: If boost is true, then you should implement boosting.
        :return: Y, a continuous numerical result
        """

        # from piazza
        #self.learners = [learner(**kwargs) for _ in range(bags)]
        self.kwargs = kwargs
        self.bags = bags
        # from hint:
        self.learners = []
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))

    # def distance(self, self.dataX, Xtest):
    #     # calculate nearest from Euclidean  distance.
    #     distance = math.sqrt((self.dataX[0]-Xtest[0])^2+(self.dataX[1]-Xtest[1])^2)
    #     return distance

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        #loop through all learners:
        for learner in self.learners:
            #take 600 data points at random from the train data with replacement
            index = np.random.choice(range(int(dataX.shape[0])),size=(dataX.shape[0]), replace=True)
            # use index to generate data and feed it into each learner
            learner.addEvidence(dataX[index], dataY[index])

    def query(self, Xtest):

        learnerOutput =  np.zeros((self.bags, len(Xtest))) # array with 'self.bags' rows and enough columns for each point (row) in x
        #loop through all learners:
        for i in range(self.bags):
            learnerOutput[i,:] = self.learners[i].query(Xtest)
        # for learner in self.learners:
        #     # append learner output to list
        #     predY = learner.query(Xtest)
        #     learnerOutput.append(predY)

        # return the means of all the learners
      # print 'BaglearnerOutput: \n', learnerOutput
      # print 'BaglearnerOutput MEAN: \n', learnerOutput.mean(axis=0)
        # print 'BaglearnerOutput Shape\n:', learnerOutput.shape()
        # print 'BaglearnerOutput MEAN Shape: \n', learnerOutput.mean(axis=0).shape()
        return learnerOutput.mean(axis=0)