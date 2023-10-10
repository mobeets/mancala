from mancala import Mancala
import numpy as np

def random_heuristic(D=15):
    return np.random.randn(D)

def initial_heuristic(D=15):
    """ returns difference in player scores """
    w = np.zeros(D)
    w[0:7] = 1. # sums player 1's score
    w[7:14] = -1. # subtracts player 2's score
    return w

def linear_heuristic(first_state, last_state, w):
    """ returns dot product of readout weights and state difference """
    return np.dot(w, np.array(last_state) - np.array(first_state))

class MonteCarloHeuristicRolloutAgent:
    """
    Monte Carlo Rollouts with a heuristic:
    - given a state, performs N rollouts starting from each available action
    and then chooses the action that led to the best average return
    - but here, each rollout is only of the next K time steps (not the full episode), and the average return is estimated as a heuristic
    - the heuristic is a function f(s,s'), where s is the current state of the game, and s' is the state of the game after the simulated rollout
    """
    def __init__(self, name=None, K=4, heuristic=None, nsamples=1000, verbose=False):
        self.name = name
        self.K = K
        if heuristic is None:
            heuristic = lambda s,snext: linear_heuristic(s, snext, initial_heuristic())
        self.heuristic = heuristic
        self.nsamples = nsamples
        self.verbose = verbose
        self.estimated_win_percents = []

    def get_rollout_policy_action(self, state):
        actions = Mancala.get_valid_actions(state)
        return np.random.choice(actions)

    def rollout(self, state, action):
        terminated = False
        i = 0
        data = [state]
        while not terminated and i < self.K:
            if i > 0:
                action = self.get_rollout_policy_action(state)
            state = Mancala.next_state(state, action)
            data.append(state)
            terminated = Mancala.is_terminated(state)
            i += 1
        return data

    def get_return(self, data, player_num):
        final_state = data[-1]
        if Mancala.is_terminated(final_state):
            winner = Mancala.get_winner(final_state)
            return 1 if winner == player_num else (0.5 if winner < 0 else 0)
        else:
            sign = 1 if (data[0][-1] == player_num) else -1
            return sign * self.heuristic(data[0], data[-1])

    def find_best_action(self, state):
        actions = Mancala.get_valid_actions(state)
        mean_payouts = []
        for action in actions:
            payouts = []
            for _ in range(self.nsamples):
                data = self.rollout(state, action)
                payout = self.get_return(data, state[-1])
                payouts.append(payout)
            mean_payouts.append(np.mean(payouts))
        return actions[np.argmax(mean_payouts)], np.array(mean_payouts)

    def get_action(self, state, index=None):
        action, probs = self.find_best_action(state)
        self.estimated_win_percents.append((index, max(probs)))
        if self.verbose:
            pct = np.round(100*max(probs),0)
            pcts = np.round(100*probs,0)
            print('CPU({}) win belief: {}% ({})'.format(self.name, pct, pcts))
        return action

def get_players(heuristics, K, nsamples):
    players = []
    for i, h in enumerate(heuristics):
        hfcn = lambda s, snext: linear_heuristic(s, snext, h)
        player = MonteCarloHeuristicRolloutAgent(name='P{}'.format(i+1), K=K, nsamples=nsamples, heuristic=hfcn)
        players.append(player)
    return players

def play(popsize, K, nsamples):
    from play import play_game
    heuristics = [initial_heuristic() if i > 0 else random_heuristic() for i in range(popsize)]
    players = get_players(heuristics, K, nsamples)
    outcomes = []
    for i in range(1,len(heuristics)):
        print('Opponent #{}'.format(i))
        outcome = play_game([players[0], players[i]], render_mode=None)
        outcomes.append(outcome)
        print(outcome)
    print(outcomes)

if __name__ == '__main__':
    play(popsize=10, K=2, nsamples=1000)
