# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #init all values to zero:
        
        
        states = self.mdp.getStates()
        
        #dont really understand why we have to do max here and set the values dictionary here
        for i in range(iterations):
          valuesCopy = self.values.copy()
          for state in states:
            finalValue = None
            for action in self.mdp.getPossibleActions(state):
              currentValue = self.computeQValueFromValues(state,action)
              if finalValue == None or finalValue < currentValue:
                finalValue = currentValue
            if finalValue == None:
              finalValue = 0
            valuesCopy[state] = finalValue
          
          self.values = valuesCopy

            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        q_value = 0
        for (succ, transition) in self.mdp.getTransitionStatesAndProbs(state,action):
            reward = self.mdp.getReward(state, action, succ)
            q_value += transition * (reward  + self.discount * self.getValue(succ))
        return q_value
        
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        if self.mdp.isTerminal(state): return None
        
        best_action = None
        best_value = float('-inf')
        for action in actions:
            qvalue = self.getQValue(state,action)
            if (qvalue > best_value):
                best_value = qvalue
                best_action = action
        return best_action    

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
