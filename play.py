import numpy as np
import matplotlib.pyplot as plt
from mancala import Mancala
from mcr import MonteCarloRolloutAgent
from mmcts import MCTSMancalaAgent

class HumanAgent:
    def get_action(self, state, index=None):
        actions = [Mancala.bin_to_letter(i) for i in Mancala.get_valid_actions(state)]
        action = -1
        while action not in actions:
            action = input("player {}'s next move? ({}): ".format(int(state[-1]), ''.join(actions))).upper()
        return Mancala.letter_to_bin(action)

def get_players(player_types, verbose, nsamples):
    players = []
    if not hasattr(nsamples, '__iter__'):
        nsamples = [nsamples]*len(player_types)
    for i, player_type in enumerate(player_types):
        if player_type == 'human':
            player = HumanAgent()
        elif player_type == 'mcr':
            player = MonteCarloRolloutAgent(name='P{}'.format(i+1), nsamples=nsamples[i], verbose=verbose)
        elif player_type == 'mcts':
            # n.b. exploitationWeight==0 is essentially a rollout algorithm
            # because we only select nodes based on avg return
            player = MCTSMancalaAgent(iterationLimit=nsamples[i]*6, exploitationWeight=0)
        players.append(player)
    return players

def plot(players, outfile):
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
    plt.savefig(outfile)

def play(player_types, nsamples, plotfile=None, verbose=True, render_mode='human'):
    env = Mancala(render_mode=render_mode, render_unicode=True)
    state, _ = env.reset()
    terminated = False

    players = get_players(player_types, verbose=verbose, nsamples=nsamples)
    while not terminated:
        action = players[int(state[-1])-1].get_action(state, index=env.index)
        state, _, terminated, _, _ = env.step(action)

        # update mcts tree
        for player in players:
            if hasattr(player, 'update'):
                player.update(action)

    if plotfile:
        plot(players, plotfile)
    return Mancala.get_winner(env.state)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=['human', 'mcr', 'mcts'])
    parser.add_argument('player2', choices=['human', 'mcr', 'mcts'])
    parser.add_argument('--nsamples', type=int, default=3000)
    parser.add_argument('--plotfile', type=str)
    args = parser.parse_args()
    play([args.player1, args.player2], nsamples=args.nsamples, plotfile=args.plotfile)
