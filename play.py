from mancala import Mancala
import numpy as np

# def sim(opening_actions, depth, check_auto_first=False, do_render=False):
#     state = initial_game_state()
#     if do_render:
#         render(state)
#         input()
#     i = 0
#     while not is_terminated(state):
#         if check_auto_first and get_auto_action(state):
#             action = get_auto_action(state)
#         elif opening_actions:
#             action = opening_actions.pop()
#             if action not in get_valid_actions(state):
#                 return None, None
#         else:
#             action = best_move(state, depth)
#         state = step(state, action)
#         if do_render:
#             render(state, action)
#             input()
#         i += 1
#     if do_render:
#         print()
#         print("player {} wins: {}".format(get_winner(state), get_current_scores(state)))
#     return get_current_scores(state), i

# def analyze_best_move(depth=2):
#     for first_action in range(6):
#         scores, nsteps = sim([first_action], depth)
#         if scores is not None:
#             score_diff = -np.diff(scores)[0]
#         else:
#             score_diff = 'DNE'
#         print(first_action, score_diff, nsteps)

# Pure Monte Carlo Rollout

class MonteCarloRolloutAgent:
    def __init__(self, name=None, nsamples=1000, verbose=False):
        self.name = name
        self.nsamples = nsamples
        self.verbose = verbose

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

    def mcts(self, state):
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

    def get_action(self, state):
        action, probs = self.mcts(state)
        if self.verbose:
            print('CPU({}) win belief: {}% ({})'.format(self.name, int(100*max(probs)), np.round(100*probs,0)))
        return action

class HumanAgent:
    def get_action(self, state):
        actions = [Mancala.bin_to_letter(i) for i in Mancala.get_valid_actions(state)]
        action = -1
        while action not in actions:
            action = input("player {}'s next move? ({}): ".format(int(state[-1]), ''.join(actions))).upper()
        return Mancala.letter_to_bin(action)

def get_players(player1, player2, verbose=False, nsamples=3000):
    if player1 == 'human':
        player1 = HumanAgent()
    else:
        player1 = MonteCarloRolloutAgent(name='P1', nsamples=nsamples, verbose=verbose)
    if player2 == 'human':
        player2 = HumanAgent()
    else:
        player2 = MonteCarloRolloutAgent(name='P2', nsamples=nsamples, verbose=verbose)
    return [player1, player2]

def play(player1, player2):
    env = Mancala(render_mode='human', render_unicode=True)
    state, _ = env.reset()
    terminated = False

    players = get_players(player1, player2, verbose=True, nsamples=3000)
    while not terminated:
        action = players[int(state[-1])-1].get_action(state)
        state, _, terminated, _, _ = env.step(action)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=['human', 'cpu'])
    parser.add_argument('player2', choices=['human', 'cpu'])
    args = parser.parse_args()
    play(args.player1, args.player2)
