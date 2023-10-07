from mancala import Mancala
import numpy as np

class MonteCarloRolloutAgent:
    def __init__(self, name=None, nsamples=1000, verbose=False):
        self.name = name
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
        while not terminated:
            if i > 0:
                action = self.get_rollout_policy_action(state)
            state = Mancala.next_state(state, action)
            data.append(state)
            terminated = Mancala.is_terminated(state)
            i += 1
        return data

    def get_return(self, data, player_num):
        final_state = data[-1]
        assert Mancala.is_terminated(final_state)
        return 1 if Mancala.get_winner(final_state) == player_num else 0

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
