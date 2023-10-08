# source: https://raw.githubusercontent.com/pbsinclair42/MCTS/master/mcts.py
from __future__ import division

import time
import math
import random

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

class treeNode:
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s = []
        s.append("totalReward: %s"%(self.totalReward))
        s.append("numVisits: %d"%(self.numVisits))
        s.append("isTerminal: %s"%(self.isTerminal))
        s.append("possibleActions: %s"%(self.children.keys()))
        return "%s: {%s}"%(self.__class__.__name__, ', '.join(s))

class MCTS:
    def __init__(self, timeLimit=None, iterationLimit=None, explorationWeight=1/math.sqrt(2), exploitationWeight=1, rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationWeight = explorationWeight
        self.exploitationWeight = exploitationWeight
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        """
        update our tree with some rollouts
        then find our best action starting from initialState
        """
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0, 1)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
        execute a round of selection-expansion-simulation-backpropagation
        """
        # selection-expansion
        node = self.selectAndExpandNode(self.root)

        # simulation
        reward = self.rollout(node.state)

        # backpropagation
        self.backpropagate(node, reward)

    def selectAndExpandNode(self, node):
        """
        performs selection and expansion steps of MCTS
            - moves down tree (following tree policy) until we reach a spot where we have not explored each possible action at least once
        """
        while not node.isTerminal:
            if node.isFullyExpanded:
                # selection
                node = self.getBestChild(node, self.explorationWeight, self.exploitationWeight)
            else:
                # expansion
                return self.expand(node)
        return node

    def expand(self, node):
        """
        find the next action that we have not taken starting from this node
        and add the state (resulting from that action) to our tree
        """
        assert not node.isFullyExpanded
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
        raise Exception("Should never have called expand() if node was fully expanded")

    def backpropagate(self, node, reward):
        """
        update node's value and then repeat for parent, until we reach root
        """
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationWeight, exploitationWeight):
        """
        of the actions available starting from node, choose an action
            based on the node's visit counts (exploration) and the node's reward value (exploitation)
        """
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeExplorationValue = math.sqrt(2*math.log(node.numVisits)/child.numVisits)
            nodeExploitationValue = node.state.getCurrentPlayer()*child.totalReward/child.numVisits
            nodeValue = explorationWeight*nodeExplorationValue + exploitationWeight*nodeExploitationValue
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)
