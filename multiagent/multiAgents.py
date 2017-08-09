# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import sys
from enum import Enum

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
        
        w1 = 100.0
        v1 = 1.0 if successorGameState.hasFood(*newPos) else 0
        
        w2 = 1.0
        v2 = 10000.0 if len([x for x in currentGameState.getCapsules() if x == newPos]) > 0 else 0
        
        w3 = 1.0
        v3 = successorGameState.getScore()
        
        w4 = .9
        v4 = sum([manhattanDistance(newPos,ghost.getPosition()) for ghost in newGhostStates])
        
        w5 = 5.0
        v5_dist = [manhattanDistance(newPos,x) for x in successorGameState.getCapsules()]
        v5 = min(v5_dist) if len(v5_dist) > 0 else 0
        v5 *= -1.0 if len(newScaredTimes) > 0 else 1
        
        w6 = 1.0
        v6 = -1000.0 if currentGameState.getPacmanPosition() == newPos else 0.0
        
        w7 = .01
        v7 = sum([manhattanDistance(newPos,d) for d in successorGameState.getFood()])
        
        w = []
        v = []
        for i in range(1,8):
            w.append(eval( "w" + str(i) ))
            v.append(eval( "v" + str(i) ))
        
        print "w", w
        print "v", v
        
        reward = dot(v,w)
        print reward
        return reward
    
    
def dot(v,w):
    if len(v) != len(w):
        raise ValueError("vectors are not the same size")
    multiply = lambda k : v[k] * w[k]
    return sum(map(multiply, range(len(v))))
    

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        currGameState = gameState
        currentDepth = 0
        maxDepth = self.depth
        currAgentIndex = 0
        maxNumOfAgents = gameState.getNumAgents() 
        action = None
        
        state = AugmentedGameState(currGameState,currentDepth,maxDepth,currAgentIndex,maxNumOfAgents,action)
        identifyAgent = self.identifyAgent
        successor = self.successorFn
        
        v = self.dispatcher(state,identifyAgent,successor)
        path = state.path()
        return path[0]
    
    def identifyAgent(self, state):
        currGameState = state.currGameState
        currentDepth = state.currentDepth
        maxDepth = state.maxDepth
        currAgentIndex = state.currAgentIndex
        maxNumOfAgents = state.maxNumOfAgents
        action = state.action
        
        if currentDepth == maxDepth: return Agent.LEAF
        elif currGameState.isWin() or currGameState.isLose() : return Agent.LEAF
        elif currAgentIndex == 0: return Agent.MAX
        else: return Agent.MIN
        
    def successorFn(self, state):
        currGameState = state.currGameState
        currentDepth = state.currentDepth
        maxDepth = state.maxDepth
        currAgentIndex = state.currAgentIndex
        maxNumOfAgents = state.maxNumOfAgents
        _ = state.action
        
        actions = currGameState.getLegalActions(currAgentIndex)
        nextAgentIndex = 0 if currAgentIndex == (maxNumOfAgents-1) else currAgentIndex+1
        nextDepth = currentDepth+1 if nextAgentIndex == 0 else currentDepth

        return [AugmentedGameState(currGameState.generateSuccessor(currAgentIndex, action),nextDepth,maxDepth,nextAgentIndex,maxNumOfAgents,action) for action in actions]
       
    def dispatcher(self, state, identifyAgent, successor):
        agent = identifyAgent(state)
        if   agent == Agent.MAX: return self.max_value(state,identifyAgent,successor)
        elif agent == Agent.MIN: return self.min_value(state,identifyAgent,successor)
        else:                    
            result = self.evaluationFunction(state.currGameState)
            return result
    
    def min_value(self, state, identifyAgent, successor):
        v = float('inf')
        next_best_game_state = None
        for s in successor(state):
            m = self.dispatcher(s,identifyAgent,successor)            
            if (m <= v):
                v = m
                next_best_game_state = s
                
            #print v
            #print "MIN",s
        state.next_best_game_state = next_best_game_state
        return v
    
    def max_value(self, state,identifyAgent,successor):
        v = float('-inf')
        next_best_game_state = None
        for s in successor(state):
            m = self.dispatcher(s,identifyAgent,successor)            
            if (m >= v):
                v = m
                next_best_game_state = s
            #print v
            #print "MAX",s
        state.next_best_game_state = next_best_game_state
        return v

class AugmentedGameState():
    def __init__(self,currGameState,currentDepth,maxDepth,currAgentIndex,maxNumOfAgents,action):
        self.currGameState = currGameState
        self.currentDepth = currentDepth
        self.maxDepth = maxDepth
        self.currAgentIndex = currAgentIndex
        self.maxNumOfAgents = maxNumOfAgents
        self.action = action
        self.next_best_game_state = None
    
    def path(self):
        acc = []
        self.buildPath(self.next_best_game_state,acc)
        return acc
                        
    def buildPath(self,state,acc):
        if (state is None): return acc
        else:
            acc.append(state.action)
            self.buildPath(state.next_best_game_state,acc)
    def __str__(self):
        desc = "augmented agent:\n"
        for k,v in vars(self).items():
            desc += str(k) + ":" + str(v) + "\n"
        return desc
        
class Agent(Enum):
    MIN = 1
    MAX = 2
    CHANCE = 3
    LEAF = 4
    
    

    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v,a = self.alphabeta(gameState, self.depth, float("-inf"), float("inf"), True, 0)
        print "value",v
        return a
    
    def isTerminal(self, gameState):
        return gameState.isWin() or gameState.isLose()
                 
    def isMaximizingPlayer(self,agentIndex):
        return agentIndex==0
    
    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        print "max_value starting"
        nextAgentIndex = agentIndex+1
        nextDepth = depth-1 if self.isMaximizingPlayer(nextAgentIndex) else nextDepth
        v = float('-inf')
        best_action = None
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIndex, action)
            m = self.alphabeta(succGameState, depth-1, alpha, beta, self.isMaximizingPlayer(nextAgentIndex),nextAgentIndex)
            #v = max(v, m[0])
            if (v < m[0]):
                v = m[0]
                best_action = action
                print "made it"
            print "value", v
            print "before alpha", alpha
            print "before beta", beta
            if (v > beta): 
                break
            alpha = max(alpha, v)
            print "after alpha", alpha
            print "after beta", beta
            print "max",action
        print "max_value ending"
        return (v,best_action)
        
    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        print "min_value starting"
        nextAgentIndex = agentIndex+1
        nextDepth = depth-1 if self.isMaximizingPlayer(nextAgentIndex) else nextDepth
        v = float('inf')
        best_action = None
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            succGameState = gameState.generateSuccessor(agentIndex, action)
            m = self.alphabeta(succGameState, depth-1, alpha, beta, self.isMaximizingPlayer(nextAgentIndex),nextAgentIndex)
            #v = min(v, m[0])
            if ( v > m[0]):
                v = m[0]
                best_action = action
                print "made it"
            print "value", v
            print "before alpha", alpha
            print "before beta", beta
            if (v < alpha): 
                break
            beta = min(beta,v)
            print "after alpha", alpha
            print "after beta", beta
            print "min",action
        print "min_value ending"
        return (v,best_action)
    
    def alphabeta(self, gameState, depth, alpha, beta, maximizingPlayer, currentAgent):
        if depth == 0 or self.isTerminal(gameState): 
            score = self.evaluationFunction(gameState)
            print "score", score
            return (score, None)
        agentIndex = currentAgent % self.depth
        if maximizingPlayer : return self.max_value(gameState,agentIndex,depth,alpha,beta)
        else : return self.min_value(gameState,agentIndex,depth,alpha,beta)
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

