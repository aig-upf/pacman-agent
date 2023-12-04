# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        
        Picks among the actions with the highest Q(s,a).
        
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions) 
        """

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = util.Counter()
        successor = self.getSuccessor(game_state, action)
        features['successorScore'] = self.getScore(successor)
        return features
    
    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)        
        self.presentCoordinates = (-5 ,-5)
        self.counter = 0
        self.attack = False
        self.lastFood = []
        self.presentFoodList = []
        self.shouldReturn = False
        self.capsulePower = False
        self.targetMode = None
        self.eatenFood = 0
        self.initialTarget = []
        self.hasStopped = 0
        self.capsuleLeft = 0
        self.prevCapsuleLeft = 0

    def registerInitialState(self, gameState):
        self.currentFoodSize = 9999999
        
        CaptureAgent.registerInitialState(self, gameState)
        self.initPosition = gameState.getAgentState(self.index).getPosition()
        self.initialAttackCoordinates(gameState)

    def initialAttackCoordinates(self ,gameState):
        
        layoutInfo = []
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x +=1
        y = (gameState.data.layout.height - 2) // 2
        layoutInfo.extend((gameState.data.layout.width , gameState.data.layout.height ,x ,y))
       
        self.initialTarget = []

        
        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.initialTarget.append((layoutInfo[2], i))
        
        noTargets = len(self.initialTarget)
        if(noTargets%2==0):
            noTargets = (noTargets//2) 
            self.initialTarget = [self.initialTarget[noTargets]]
        else:
            noTargets = (noTargets-1)//2
            self.initialTarget = [self.initialTarget[noTargets]] 

    
    def evaluateAttackParameters(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action) 
        position = successor.getAgentState(self.index).getPosition() 
        foodList = self.getFood(successor).asList() 
        features['successorScore'] = self.getScore(successor) 

        if successor.getAgentState(self.index).isPacman:
            features['offence'] = 1
        else:
            features['offence'] = 0
        if foodList: 
            features['foodDistance'] = min([self.getMazeDistance(position, food) for food in foodList])

        opponentsList = []
        disToGhost = []
        opponentsList = self.getOpponents(successor)

        for i in range(len(opponentsList)):
            enemyPos = opponentsList[i]
            enemy = successor.getAgentState(enemyPos)
            if not enemy.isPacman and enemy.getPosition() != None:
                ghostPos = enemy.getPosition()
                disToGhost.append(self.getMazeDistance(position ,ghostPos))

        if len(disToGhost) > 0:
            minDisToGhost = min(disToGhost)
            if minDisToGhost < 5:
                features['distanceToGhost'] = minDisToGhost + features['successorScore']
            else:
                features['distanceToGhost'] = 0
        return features
    
    def getCostOfAttackParameter(self, gameState, action):
        '''
        Setting the weights manually after many iterations
        '''
        if self.attack:
            if self.shouldReturn is True:
                return {'offence' :3010,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
            else:
                return {'offence' :0,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
        else:
            successor = self.getSuccessor(gameState, action) 
            weightGhost = 210
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            if len(invaders) > 0:
                if invaders[-1].scaredTimer > 0:
                    weightGhost = 0
                    
            return {'offence' :0,
                    'successorScore': 202,
                    'foodDistance': -8,
                    'distancesToGhost' :weightGhost}

    def getOpponentPositions(self, gameState):
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def bestPossibleAction(self ,mcsc):
        ab = mcsc.getLegalActions(self.index)
        ab.remove(Directions.STOP)

        if len(ab) == 1:
            return ab[0]
        else:
            reverseDir = Directions.REVERSE[mcsc.getAgentState(self.index).configuration.direction]
            if reverseDir in ab:
                ab.remove(reverseDir)
            return random.choice(ab)

    def monteCarloSimulation(self ,gameState ,depth):
        ss = gameState.deepCopy()
        while depth > 0:
            ss = ss.generateSuccessor(self.index ,self.bestPossibleAction(ss))
            depth -= 1
        return self.evaluate(ss ,Directions.STOP)

    def getBestAction(self,legalActions,gameState,possibleActions,distanceToTarget):
        shortestDistance = 9999999999
        for i in range (0,len(legalActions)):    
            action = legalActions[i]
            nextState = gameState.generateSuccessor(self.index, action)
            nextPosition = nextState.getAgentPosition(self.index)
            distance = self.getMazeDistance(nextPosition, self.initialTarget[0])
            distanceToTarget.append(distance)
            if(distance<shortestDistance):
                shortestDistance = distance

        bestActionsList = [a for a, distance in zip(legalActions, distanceToTarget) if distance == shortestDistance]
        bestAction = random.choice(bestActionsList)
        return bestAction
        
    def chooseAction(self, gameState):
        self.presentCoordinates = gameState.getAgentState(self.index).getPosition()
    
        if self.presentCoordinates == self.initPosition:
            self.hasStopped = 1
        if self.presentCoordinates == self.initialTarget[0]:
            self.hasStopped = 0

        # To find the best possible move 
        if self.hasStopped == 1:
            legalActions = gameState.getLegalActions(self.index)
            legalActions.remove(Directions.STOP)
            distToTarget = []
            possibleActions = []
            bestAction = self.getBestAction(legalActions,gameState,possibleActions,distToTarget)
            return bestAction

        if self.hasStopped==0:
            self.presentFoodList = self.getFood(gameState).asList()
            self.capsuleLeft = len(self.getCapsules(gameState))
            realLastCapsuleLen = self.prevCapsuleLeft
            realLastFoodLen = len(self.lastFood)

            # returned = 1 when the pacman has food and should return home           
            if len(self.presentFoodList) < len(self.lastFood):
                self.shouldReturn = True
            self.lastFood = self.presentFoodList
            self.prevCapsuleLeft = self.capsuleLeft

            if not gameState.getAgentState(self.index).isPacman:
                self.shouldReturn = False

            # To check the attack situation           
            remainingFoodList = self.getFood(gameState).asList()
            remainingFoodSize = len(remainingFoodList)
    
            if remainingFoodSize == self.currentFoodSize:
                self.counter = self.counter + 1
            else:
                self.currentFoodSize = remainingFoodSize
                self.counter = 0
            if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
                self.counter = 0
            if self.counter > 20:
                self.attack = True
            else:
                self.attack = False
            
            actionsBase = gameState.getLegalActions(self.index)
            actionsBase.remove(Directions.STOP)

            # Distance to the closest enemy        
            distToEnemy = 999999
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer == 0]
            if len(invaders) > 0:
                distToEnemy = min([self.getMazeDistance(self.presentCoordinates, a.getPosition()) for a in invaders])
            
            '''
            Capsule eating:
            - capsulePower is True if there is capsule available
            - capsulePower is False if enemy Distance is less than 5.
            - capsulePower is False if a pacman has returned a food home
            '''

            if self.capsuleLeft < realLastCapsuleLen:
                self.capsulePower = True
                self.eatenFood = 0
            if distToEnemy <= 5:
                self.capsulePower = False
            if (len(self.presentFoodList) < len (self.lastFood)):
                self.capsulePower = False
            if self.capsulePower:
                if not gameState.getAgentState(self.index).isPacman:
                    self.eatenFood = 0
                modeMinDist = 999999
                if len(self.presentFoodList) < realLastFoodLen:
                    self.eatenFood += 1
                if len(self.presentFoodList )==0 or self.eatenFood >= 5:
                    self.targetMode = self.initPosition
                else:
                    for food in self.presentFoodList:
                        dist = self.getMazeDistance(self.presentCoordinates ,food)
                        if dist < modeMinDist:
                            modeMinDist = dist
                            self.targetMode = food

                legalActions = gameState.getLegalActions(self.index)
                legalActions.remove(Directions.STOP)
                possibleActions = []
                distToTarget = []
                k=0
                while k!=len(legalActions):
                    a = legalActions[k]
                    newpos = (gameState.generateSuccessor(self.index, a)).getAgentPosition(self.index)
                    possibleActions.append(a)
                    distToTarget.append(self.getMazeDistance(newpos, self.targetMode))
                    k+=1
        
                minDis = min(distToTarget)
                bestActions = [a for a, dis in zip(possibleActions, distToTarget) if dis== minDis]
                bestAction = random.choice(bestActions)
                return bestAction
            
            else:
                self.eatenFood = 0
                distToTarget = []
                for a in actionsBase:
                    nextState = gameState.generateSuccessor(self.index, a)
                    value = 0
                    for i in range(1, 24):
                        value += self.monteCarloSimulation(nextState ,20)
                    distToTarget.append(value)

                best = max(distToTarget)
                bestActions = [a for a, v in zip(actionsBase, distToTarget) if v == best]
                bestAction = random.choice(bestActions)
            return bestAction


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.previousFood = []
        self.counter = 0

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.setPatrolPoint(gameState)

    def setPatrolPoint(self ,gameState):
        
        #To look for the center of the maze for patrolling
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        self.patrolPoints = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(x, i):
                self.patrolPoints.append((x, i))

        for i in range(len(self.patrolPoints)):
            if len(self.patrolPoints) > 2:
                self.patrolPoints.remove(self.patrolPoints[0])
                self.patrolPoints.remove(self.patrolPoints[-1])
            else:
                break
    
    def getNextDefensiveMove(self ,gameState):
        agentActions = []
        actions = gameState.getLegalActions(self.index)
        rev_dir = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        actions.remove(Directions.STOP)

        for i in range(0, len(actions)-1):
            if rev_dir == actions[i]:
                actions.remove(rev_dir)

        for i in range(len(actions)):
            act = actions[i]
            newState = gameState.generateSuccessor(self.index, act)
            if not newState.getAgentState(self.index).isPacman:
                agentActions.append(act)
        
        if len(agentActions) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
        if self.counter > 4 or self.counter == 0:
            agentActions.append(rev_dir)
        return agentActions

    def chooseAction(self, gameState):
        pos = gameState.getAgentPosition(self.index)
        if pos == self.target:
            self.target = None
        invaders = []
        nearestInvader = []
        minDistance = float("inf")

        # To look for an enemy position in our home        
        opponentsPositions = self.getOpponents(gameState)
        i = 0
        while i != len(opponentsPositions):
            opponentPos = opponentsPositions[i]
            opponent = gameState.getAgentState(opponentPos)
            if opponent.isPacman and opponent.getPosition() != None:
                opponentPos = opponent.getPosition()
                invaders.append(opponentPos)
            i = i + 1

        # If there's an enemy in our base, kill it
        if len(invaders) > 0:
            for oppPos in invaders:
                dist = self.getMazeDistance(oppPos, pos)
                if dist < minDistance:
                    minDistance = dist
                    nearestInvader.append(oppPos)
            self.target = nearestInvader[-1]

        # If an enemy has food, then we remove it from targets
        else:
            if len(self.previousFood) > 0:
                if len(self.getFoodYouAreDefending(gameState).asList()) < len(self.previousFood):
                    yummy = set(self.previousFood) - set(self.getFoodYouAreDefending(gameState).asList())
                    self.target = yummy.pop()

        self.previousFood = self.getFoodYouAreDefending(gameState).asList()
        if self.target == None:
            if len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
                highPriorityFood = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
                self.target = random.choice(highPriorityFood)
            else:
                self.target = random.choice(self.patrolPoints)
        candAct = self.getNextDefensiveMove(gameState)
        bestMoves = []
        fvalues = []
        i=0
        # To find the best move       
        while i < len(candAct):
            a = candAct[i]
            nextState = gameState.generateSuccessor(self.index, a)
            newpos = nextState.getAgentPosition(self.index)
            bestMoves.append(a)
            fvalues.append(self.getMazeDistance(newpos, self.target))
            i = i + 1

        best = min(fvalues)
        bestActions = [a for a, v in zip(bestMoves, fvalues) if v == best]
        bestAction = random.choice(bestActions)
        return bestAction
