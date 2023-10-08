import random
from copy import deepcopy
from mancala import Mancala
from mcts import mcts

class MCTSMancalaNode:
    def __init__(self):
        self.state = Mancala.initial_game_state()

    def getCurrentPlayer(self):
        return -(2*(self.state[-1]-1)-1)

    def getPossibleActions(self):
        return Mancala.get_valid_actions(self.state)

    def takeAction(self, action):
        new_node = deepcopy(self)
        new_node.state = Mancala.next_state(new_node.state, action)
        return new_node

    def isTerminal(self):
        return Mancala.is_terminated(self.state)

    def getReward(self):
        winner = Mancala.get_winner(self.state) # 0 means tie
        return -(2*(winner-1)-1) if winner != 0 else 0

class MCTSMancalaAgent:
    def __init__(self, timeLimit=1000):
        self.node = MCTSMancalaNode()
        self.searcher = mcts(timeLimit=timeLimit)

    def update(self, action):
        self.node = self.node.takeAction(action)

    def get_action(self, state, index=None):
        assert state == [int(x) for x in self.node.state]
        return self.searcher.search(initialState=self.node)
