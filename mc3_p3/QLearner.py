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

        #  # create a new matrix T count
        # #self.Tc = np.empty(shape=(num_states, num_actions, num_states))
        # self.Tc = np.zeros(shape=(num_actions, num_states, num_states))
        #
        # self.Tc.fill(0.000001)
        # #self.Tc.fill(1./self.num_states)
        #
        #
        # # print 'sum Tc', sum(self.Tc[self.a, self.s, :])
        # self.T = np.zeros(shape=(num_actions, num_states, num_states))
        # #self.T.fill(1./self.num_states)
        # self.T.fill(0.000001)
        # self.R = np.zeros(shape=(num_states, num_actions))

        # should reserve space for keeping track of Q[s, a] for the number of states and actions. It should initialize Q[] with uniform random values between -1.0 and 1.0.
        self.Q = np.random.uniform(low=-1.0, high=1.0, size=(num_states, num_actions))

        # record experience tuples
        self.tupleList = []



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

        # record experience tuple
        self.tupleList.append([self.s,self.a,s_prime, r])

        # DYNA

        if self.dyna > 0:

            # # increment Tc
            # self.Tc[self.a, self.s, s_prime] += 1
            #
            # # compute T[s,a,s'] from Tc
            # # TODO: double check if should be instead:
            # #self.T[self.a, self.s , s_prime] = self.Tc[self.a, self.s , s_prime] / np.sum((self.Tc[self.a,self.s,:]))
            # self.T[self.a, self.s , :] = self.Tc[self.a, self.s , :] / np.sum((self.Tc[self.a,self.s,:]))
            # # print 'sum(self.T[a,s,:])\n', sum(self.T[self.a,self.s,:])
            #
            # # update R[self.s,self.a]
            # self.R[self.s,self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            for n in range(0, self.dyna):

                # randomly sample from the experience tuple list

                randomTuple = rand.sample(self.tupleList, 1)
                dyna_s, dyna_a, dyna_s_prime, dyna_r = randomTuple[0]
                # print 'dyna_s: ', dyna_s

                # Update Q with dyna values (s,a,s',r)
                self.Q[dyna_s, dyna_a] = (1-self.alpha) * self.Q[dyna_s, dyna_a] \
                    + self.alpha * (dyna_r + self.gamma * self.Q[dyna_s_prime, np.argmax(self.Q[dyna_s_prime,:])])

                # # 1. randomly select s
                #
                # dyna_s = np.random.randint(0, self.num_states)
                #
                # # 2. randomly select a
                #
                # dyna_a = np.random.randint(0, self.num_actions)
                #
                # # 3 consult T to find s'
                # #   use a for loop to accumulate the sum of the probabilities as you iterate over all possible s'
                # # s_prime_dyna = 0
                # # randomNum = np.random.random()
                # # sum_prob = 0
                # # for s_prime_loop in range(self.num_states):
                # #     sum_prob += self.T[a,s,s_prime_loop]
                # #     if sum_prob > randomNum:
                # #         s_prime_dyna = s_prime_loop
                # #         break
                #
                # dyna_s_prime = np.random.multinomial(1, self.T[dyna_a, dyna_s]).argmax()
                #
                #
                # # 4. infer reward: r = R[s,a]
                # dyna_r = self.R[dyna_s,dyna_a]
                #
                # # Update Q with dyna values (s,a,s',r)
                # self.Q[dyna_s, dyna_a] = (1-self.alpha) * self.Q[dyna_s, dyna_a] \
                #     + self.alpha * (dyna_r + self.gamma * self.Q[dyna_s_prime, np.argmax(self.Q[dyna_s_prime,:])])



        # choose action according to Q table:  self.Q[s_prime,:]

        # allow potential for random action by looking at rar

         # 1. throw dice to see if random choice:
        dice = np.random.random() # random # b/w 0 and 1
        # if yes:
        if dice < self.rar:
            action = np.random.randint(0, self.num_actions)
        else:
            #select highest Q value
            action = np.argmax(self.Q[s_prime,:], axis = 0)

        # decary rar
        self.rar *= self.radr

        self.s = s_prime
        self.a = action

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
