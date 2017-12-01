from GridWorld import GridWorld
from copy import copy

class QTester(object):
    ''' tester for gridworld with learner
    '''
    def __init__(self, world, learner, maxTestIter=100):
        ''' instantiates a tester
        PARAMETERS
        ----------
        world (GridWorld): GridWorld object
        learner (QLearner): QLearner object
        maxTestIter (int): max threshold for trying to get a policy out of the learner
        '''
        self.learner = learner
        self.world = world
        self.n = 0
        self.state = None
        self.reward = None
        self.maxIter = maxTestIter
        self.action2SymbolDict = {0: '^',
                                  1: '>',
                                  2: 'v', 
                                  3: '<',
                                  '*': '*'}
        
    def iterStep(self):
        ''' runs a single experience tuple iteration
        RETURNS
        -------
        action (int): action index
        '''
        if self.n == 0:
            action = self.learner.querysetstate(tuckerHash(self.world.start))
            self.state = self.world.start
        else:
            action = self.learner.query(tuckerHash(self.state), self.reward)
        self.state, self.reward = self.world.getNewStateReward(self.state, action)
        self.n += 1
        return action
        
    def nIter(self, n):
        ''' runs n experience steps
        PARAMETERS
        ----------
        n (int): number of steps to run
        '''
        for i in range(n):
            self.iterStep()

    def getPolicy(self):
        ''' starts at the world begining and records actions till the goal state is hit

        this function kinda sucks. It assumes all sorts of things about the world.
        unfortunately instead of making a checkGoalState function in the world,
        I made it teleport back to start with a big reward... oh well. 
        I'd do it differently next time.
        '''
        actionList = []
        nIter = 0
        self.state = self.world.start
        action = self.learner.querysetstate(tuckerHash(self.state))
        while nIter < self.maxIter:
            self.state, self.reward = self.world.getNewStateReward(self.state, action)
            actionList.append(action)
            nIter += 1
            if self.reward == self.world.goalReward:
                actionList.append('*')
                break
            action = self.learner.query(tuckerHash(self.state), self.reward)
        return [self.action2SymbolDict[a] for a in actionList]

def tuckerHash(coord):
    return coord[0]*10 + coord[1]

def baseTester():
    ''' runs a somewhat comprehensive test'''
    try:
        import QLearner as ql
    except:
        pass

    #it is worth noting here that num_states can be 100 for any grid < 10x10 using the tuckerHash
    #we need a new hash algo if we are to use a grid outside those parameters
    baseKwargs = {'num_states':100, 'alpha':1.0, 'gamma':0.9, 'rar':0.5, 'radr':0.99, 'dyna':0, 'verbose':False}
    '''
    if you want to add your own test, add it here. I use a tuple to indicate one test it is:
    (csv file, expected convergence iterations, kwarg modifier, test name)
    '''
    myTestList = [('testEasyWorld.csv', 800, 13,{}, 'easy test'),
                  ('world01.csv', 7000, 16, {}, 'Tucker Test 1'),
                  ('world02.csv', 7000, 17, {}, 'Tucker Test 2'),
                  ('testGridWorld.csv', 5000, 20, {}, 'Leo Base Test'),
                  ('testGridWorld.csv', 18000, 20, {'alpha':.2}, 'Test Learning Rate'),
                  ('testEasyWorld.csv', 700, 13, {'rar': 0.05}, 'Test Exploration'),
                  ('testEasyWorld.csv', 700, 13, {'radr': 0.8}, 'Test Exploration Decay'),
                  ('testGridWorld.csv', 3000, 20, {'gamma':0.8}, 'Test Discount Rate'),
                  ('testGridWorld.csv', 1100, 20, {'dyna':100}, 'Test Dyna'),
                  ]
    for test in myTestList:
        print '-------------------------------'
        print test[4]
        world = GridWorld(test[0])
        testKwargs = copy(baseKwargs)
        for k in test[3].keys():
            testKwargs[k] = test[3][k]
        print 'parameters %s' % str(testKwargs)
        learner = ql.QLearner(**testKwargs)
        print world.grid
        myTester = QTester(world, learner)
        nIter = test[1]
        totalIter = nIter
        lastPolicyLength = 0
        #someone let me know if there's a better way to check for convergence time
        while (totalIter < (test[1] * 1.4)):
           myTester.nIter(nIter)
           nIter = int(.05*test[1])
           myPolicy = myTester.getPolicy()
           policyLength = len(myPolicy)
           totalIter += nIter
           if (lastPolicyLength == policyLength) and (policyLength < 100):
              print 'converged in approx %i iterations' % totalIter
              print policyLength, myPolicy
              break
           lastPolicyLength = policyLength
        if (test[1]*1.2 >= totalIter) and (policyLength == test[2]):
           print '*** TEST PASSED ***'
        else:
           print 'xxx TEST FAILED xxx'

if __name__=='__main__':
    baseTester()
