from mancala import Mancala
import numpy as np
import matplotlib.pyplot as plt
from agent import MonteCarloRolloutAgent
from mmcts import MCTSMancalaAgent

class HumanAgent:
    def get_action(self, state, index=None):
        actions = [Mancala.bin_to_letter(i) for i in Mancala.get_valid_actions(state)]
        action = -1
        while action not in actions:
            action = input("player {}'s next move? ({}): ".format(int(state[-1]), ''.join(actions))).upper()
        return Mancala.letter_to_bin(action)

def get_players(player1, player2, verbose, nsamples):
    if player1 == 'human':
        player1 = HumanAgent()
    elif player1 == 'mcrollout':
        player1 = MonteCarloRolloutAgent(name='P1', nsamples=nsamples, verbose=verbose)
    else:
        player1 = MCTSMancalaAgent(timeLimit=nsamples)
    if player2 == 'human':
        player2 = HumanAgent()
    elif player2 == 'mcrollout':
        player2 = MonteCarloRolloutAgent(name='P2', nsamples=nsamples, verbose=verbose)
    else:
        player2 = MCTSMancalaAgent(timeLimit=nsamples)
    return [player1, player2]

def plot(players):
    import matplotlib.pyplot as plt
    if hasattr(players[0], 'estimated_win_percents'):
        pts1 = np.vstack(players[0].estimated_win_percents)
        plt.plot(pts1[:,0], 100*pts1[:,1], '.-', label='P1')
    if hasattr(players[1], 'estimated_win_percents'):
        pts2 = np.vstack(players[1].estimated_win_percents)
        plt.plot(pts2[:,0], 100*pts2[:,1], '.-', label='P2')
    plt.xlabel('move index')
    plt.ylabel('estimated win percent')
    plt.ylim([0, 100])
    plt.legend()
    plt.savefig('plots/tmp.png')

def play(player1, player2, nsamples, verbose=True):
    env = Mancala(render_mode='human', render_unicode=True)
    state, _ = env.reset()
    terminated = False

    players = get_players(player1, player2, verbose=verbose, nsamples=nsamples)
    while not terminated:
        action = players[int(state[-1])-1].get_action(state, index=env.index)
        state, _, terminated, _, _ = env.step(action)

        # update mcts tree
        for player in players:
            if hasattr(player, 'update'):
                player.update(action)

    # plot(players)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=['human', 'mcrollout', 'mcts'])
    parser.add_argument('player2', choices=['human', 'mcrollout', 'mcts'])
    parser.add_argument('--nsamples', type=int, default=3000)
    args = parser.parse_args()
    play(args.player1, args.player2, nsamples=args.nsamples)
