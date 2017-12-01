from pandas import DataFrame
import numpy as np

class GridWorld(object):
    ''' GridWorld takes care of all the world dynamics for a non-stochastic grid world'''
    def __init__(self, csvFile):
        ''' instantiates a GridWorld
        PARAMETERS
        ----------
        csvFile (str): path to a csv file to be parsed into string
                       grid uses the following key:
                       0 empty space
                       1 wall
                       2 start 
                       3 goal 
        '''
        self.grid = DataFrame.from_csv(csvFile, header=None, index_col=None).values
        self.start = np.array(np.where(self.grid == 2)).flatten()
        self.end = np.array(np.where(self.grid == 3)).flatten()
        self.actionDict = {0: np.array([-1, 0]),
                           1: np.array([0, 1]),
                           2: np.array([1, 0]),
                           3: np.array([0, -1])} 
        #not sure if these are the same as Tucker, but it doens't matter since learner is world independent
        self.stepPenalty = -1.
        self.goalReward = 5.
        
    def getNewStateReward(self, state, action):
        ''' gets world to give the new state from a state action pair
        PARAMETERS
        ----------
        state (np.array): numpy array coordinate
        action (int): one of the viable actions in self.actionDict
        RETURNS
        -------
        state, reward (np.array, float): state coordinate and reward value
        '''
        if action not in self.actionDict.keys():
            raise("Warning action is not an acceptable action key")
        newState = state + self.actionDict[action]

        #check if off grid in negative direction 
        if any(newState < 0):
            return state, self.stepPenalty
        #check if off grid in positive direction
        elif any([newState[i] > (self.grid.shape[i]-1) for i in range(len(self.grid.shape))]):
            return state, self.stepPenalty
        #in hindsight could have used mod to make a looping world with fewer lines of code
        #check for wall
        elif self.grid[tuple(newState)] == 1:
            return state, self.stepPenalty
        #check for goal
        elif self.grid[tuple(newState)] == 3:
            return self.start, self.goalReward
        #otherwise just move
        else:
            return newState, self.stepPenalty
