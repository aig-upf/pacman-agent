# baseline_team.py
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random

import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
import pickle

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

class QLearningAgent(CaptureAgent):

    """
    Our agent that uses QLearning to find the best possible move
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.q_values = util.Counter()

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        """
        #load q-values from pickle file if they exist
        try:
            with open("q_values.pkl", "rb") as f:
                self.q_values = pickle.load(f)
        except FileNotFoundError:
            self.q_values = util.Counter()
        """


    def getQValue(self, game_state, action):
        return self.q_values[(game_state,action)]

    def computeValue(self, game_state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get all legal actions for the given state.
        legal_actions = self.getLegalActions(game_state)

        # If there are no legal actions return 0
        if len(legal_actions) == 0:
            return 0.0

        # The value is just the max of all q values.
        value = float("-inf")
        for action in legal_actions:
            # Compute the q value for each action
            q_value = self.getQValue(game_state, action)
            # Update the value to be the max of the q values.
            value = max(value, q_value)

        return value

    def find_best_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)


        legal_actions = game_state.get_legal_actions(self.index)

        best_action = None
        max_q_value = float("-inf")

        # If it's a terminal state.
        if len(legal_actions) == 0:
            return best_action

        for action in legal_actions:
            q_value = self.getQValue(game_state, action)
            # If the q_value is higher than current max-q, update max-q and best action.
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
            elif q_value == max_q_value:
                best_action = random.choice([best_action, action])
                # If the action chosen is action and not best_action,
                # update the max_q_value to be the q_value of action
                if action == best_action:
                    max_q_value = q_value

        return best_action

    def choose_best_action(self, game_state):
        # Pick Action
        legalActions = self.getLegalActions(game_state)
        action = None
        "*** YOUR CODE HERE ***"
        # Check for terminal state.
        if len(legalActions) == 0:
            return action  # At this moment action = None

        prob = self.epsilon  # Probability to take a random action
        if util.flipCoin(prob):  # If True, take a random action
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(game_state)

        return action

    def update(self, game_state, action, next_game_state, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        q_value = self.getQValue(game_state, action)
        alpha = self.alpha
        discount = self.discount
        # The value of the next state = argmax of the Q-values of the next state.
        value_next_state = self.computeValue(next_game_state)

        # Update the q value in the dictionary q_values. Just the formula from the slides CS188.
        self.q_values[game_state, action] = ((1 - alpha) * q_value) + \
                                        (alpha * (reward + (discount * value_next_state)))

class OffensiveQLearningAgent(QLearningAgent):
    """

    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):

        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, game_state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,game_state)
        self.doAction(game_state,action)
        return action

    """
    def final(self, game_state):

        # Saves Q-values at the end of every game
        with open("q_values.pkl", "wb") as f:
            pickle.dump(dict(self.q_values), f)
    """