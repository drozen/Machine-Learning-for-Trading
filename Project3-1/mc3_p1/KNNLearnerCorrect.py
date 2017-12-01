"""
KNN LEARNER by Daniel Rozen
"""
import math
import numpy as np

class KNNLearner(object):

    def __init__(self, k):
        if k == 0:
            raise Exception("k must be greater than 0")
        self.k = k

    def distance(self, X1, X2):
        # calculate nearest from Euclidean  distance.
        distance = math.sqrt((X1[0]-X2[0])^2+(X1[1]-X2[1])^2)
        return distance

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        #just store the train and test data supplied in that step, so it is available for the query step
        self.dataX = dataX
        self.dataY = dataY

    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # build and save the model:


        # for i in range(len(dataX)): # loop through every potential data point
        print 'len(Xtest) \n', len(Xtest)
        print 'len(self.dataX) \n', len(self.dataX)

        distances = np.zeros((len(self.dataX), len(Xtest)))

        X1 = self.dataX
        X2 = Xtest
        for i in range(len(Xtest)):
            results = np.zeros(len(self.dataX))
            for j in range((len(self.dataX))):
                distances[i,j] = math.sqrt((X1[i, 0]-X2[j, 0])**2+(X1[i, 1]-X2[j, 1])**2)
                #distances[i,j] = math.sqrt((X1(i, 0)-X2(j, 0))**2+(X1(i, 1)-X2(j, 1))**2);

                #distances[i,j] = math.sqrt((X1[0, i]-X2[0, j])**2+(X1[1, i]-X2[1, j])**2)

            # Start with an empty array of 'k' values and store the first 'k' distances in it.

            sortedIndices = np.argsort(distances)
            #print 'indices: \n', indices

            kInstances = np.zeros(self.k)
            for n in range(self.k):
                kInstances[n] = self.dataY[sortedIndices[n]]
            results[i] = np.mean(kInstances)
            #print 'results: \n', results
        #print 'results: \n', results
        return results

            # Now, go through your remaining distances.
           # for X in dataX:

        # If you find a distance smaller than stored in your array, replace the biggest value in the array with the smaller distance.

        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataYlassify

        # Use Euclidean distance. Take the mean of the closest k points' Y values to make your prediction. If there are multiple equidistant points on the boundary of being selected or not selected, you may use whatever method you like to choose among them.

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"