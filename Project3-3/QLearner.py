"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    # use the following numbers that are distributed with the code
    # radr = random action decay rate
    # when testing, verbose will be set to false, be sure nothing prints when verbose = false
    # dyna = how many dyna updates you should do for each update
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.98, \
        radr = 0.999, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        #my additions:
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Tc = np.empty(shape=((num_states, num_actions)))                # create a new matrix T count
        self.Tc.fill(0.00001)
        self.T = self.Tc

        # should reserve space for keeping track of Q[s, a] for the number of states and actions. It should initialize Q[] with uniform random values between -1.0 and 1.0.
        self.Q = np.random.uniform(low=-1.0, high=1.0, size=(num_states, num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        Set state to s, return action for state s, but don't update Q-table.
        A special version of the query method that sets the state to s, and returns an integer action according to the same rules as query(), but it does not execute an update to the Q-table. This method is typically only used once, to set the initial state.

        Use when you don't know what your initial state.  It saves the state you set.
        @param s: The new state
        @returns: The selected action
        """
        self.s = s # Set state to s

        action = np.argmax(self.Q[self.s,:], axis = 0)

        if self.verbose: print "s =", s,"a =",action

        self.a = action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        the core method of the Q-Learner. It should keep track of the last state s and the last action a, then use the new information s_prime and r to update the Q table. The learning instance, or experience tuple is <s, a, s_prime, r>. query() should return an integer, which is the next action to take. Note that it should choose a random action with probability rar, and that it should update rar according to the decay rate radr at each step. Details on the arguments:
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # #keep track of the last state s and the last action a
        # self.s_last = self.s
        # self.a_last = self.a

        # update Q table:

        self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] \
            + self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime,:])])

        # choose action according to Q table:  self.Q[s_prime,:]

        # allow potential for random action by looking at rar

         # 1. throw dice to see if random choice:
        dice = np.random.random() # random # b/w 0 and 1
        # if yes:
        if dice < self.rar:
            action = np.random.randint(0, self.num_actions-1)
        else:
            #select highest Q value
            action = np.argmax(self.Q[s_prime,:], axis = 0)

        # decary rar
        self.rar *= self.radr

        self.s = s_prime
        self.a = action

        #    # DYNA
        #
        # if self.dyna > 0:
        #
        #     # 1. randomly select s
        #
        #     s = np.random.random() * 100
        #
        #     # 2. randomly select a
        #
        #     a = np.random.randint(0, self.num_actions-1)
        #
        #     # 3 consult T to find s'
        #     #   use a for loop to accumulate the sum of the probabilities as you iterate over all possible s'
        #
        #     randomNum = np.random.random()
        #     sum_prob = 0
        #     for s_prime_loop in range(self.num_states):
        #         sum_prob += self.T[s,a,s_prime]
        #         if sum_prob > randomNum:
        #             s_prime = s_prime_loop

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
