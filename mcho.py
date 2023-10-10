import numpy as np
from cmaes import CMA
from mancala import Mancala
from play import play_game
from mch import linear_heuristic, MonteCarloHeuristicRolloutAgent

def get_players(heuristics, K, nsamples):
    players = []
    for i, h in enumerate(heuristics):
        hfcn = lambda s, snext: linear_heuristic(s, snext, h)
        player = MonteCarloHeuristicRolloutAgent(name='P{}'.format(i+1), K=K, nsamples=nsamples, heuristic=hfcn)
        players.append(player)
    return players

def round_robin(players, verbose=False, render_mode=None):
    scores = np.nan * np.ones((len(players), len(players)))
    for i,p1 in enumerate(players):
        for j,p2 in enumerate(players):
            if verbose:
                print('{} vs. {}'.format(p1.name, p2.name))
            ps = [p1, p2]
            outcome = play_game(ps, render_mode=render_mode)
            score = 1 if outcome == 1 else (0 if outcome == 2 else 0.5)
            scores[i,j] = score
            if verbose:
                print('Outcome: {}'.format(score))
        if verbose:
            print('{} avg outcome: {}'.format(p1.name, scores[i].mean()))
    return scores

def learn(nsteps=50, K=2, nsamples=500, verbose=True):
    # todo: if nsamples > total nodes possible in K steps, we should be methodical    
    optimizer = CMA(mean=np.zeros(15), sigma=1.3)
    for generation in range(nsteps):
        players = []
        hs = []
        for i in range(optimizer.population_size):
            h = optimizer.ask()
            hfcn = lambda s, snext: linear_heuristic(s, snext, h)
            player = MonteCarloHeuristicRolloutAgent(name='P{}'.format(i+1), K=K, nsamples=nsamples, heuristic=hfcn)
            players.append(player)
            hs.append(h)
        scores = round_robin(players, verbose=verbose)
        scores = scores.mean(axis=0)
        print(generation, scores)

        solutions = list(zip(hs, scores))
        optimizer.tell(solutions)

if __name__ == '__main__':
    learn()
