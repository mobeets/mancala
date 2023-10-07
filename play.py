from mancala import Mancala
import numpy as np

def estimate_value(data, player_num):
    # data is list of (state, action, next_state)
    # we only use current state to figure out whose turn it is
    # and also check if game is over, in which case we need no heuristic
    state = data[-1][2]
    if Mancala.is_terminated(state):
        # no heuristic needed: value is inf if we won, -inf if we lost
        return np.inf if Mancala.get_winner(state) == player_num else -np.inf
    else:
        start_state = data[0][0]
        s1_init, s2_init = Mancala.get_current_scores(start_state)
        s1_final, s2_final = Mancala.get_current_scores(state)
        # value is equal to our improvement in score differential
        score_delta = (s1_final - s2_final) - (s1_init - s2_init)
        return score_delta if start_state[-1] == 1 else -score_delta

def get_auto_action(state):
    # if any action leads to an extra move, we make them here
    actions = Mancala.get_valid_actions(state)
    possible_auto_actions = []
    for action in actions:
        next_state = Mancala.next_state(state, action)
        if next_state[-1] == state[-1]:
            # taking this action would not change whose turn it is
            possible_auto_actions.append(action)
    if possible_auto_actions:
        # take the auto action closest to our mancala first
        # to ensure we can take the others on our next turn
        return max(possible_auto_actions)
    else:
        return None

def rollout_best_move(state, action, depth, nsamples):
    data = []
    for d in range(depth+1):
        if d > 0:
            action = best_move(state, depth-d, nsamples)        
        next_state = Mancala.next_state(state, action)
        data.append((state, action, next_state))
        state = next_state
    return data

def best_move(state, depth, nsamples=1):
    # first check for any automatic moves, and add them here
    chosen_action = get_auto_action(state)
    if chosen_action:
        return chosen_action

    # now check for best next move
    actions = Mancala.get_valid_actions(state)
    values = []
    for action in actions:
        cvalues = []
        for _ in range(nsamples):
            data = rollout_best_move(state, action, depth, nsamples)
            value = estimate_value(data, state[-1])
            cvalues.append(value)
        values.append(np.mean(cvalues))
    return actions[np.argmax(values)]

class HeuristicAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def get_action(self, state):
        return best_move(state, self.depth)

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

# MCTS

class MCTSAgent:
    def __init__(self, nsamples=1000):
        self.nsamples = nsamples

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
        return actions[np.argmax(mean_payouts)], np.max(mean_payouts)

    def get_action(self, state):
        action, prob = self.mcts(state)
        print('CPU win belief: {}%'.format(int(100*prob)))
        return action

class HumanAgent:
    def get_action(self, state):
        actions = [Mancala.bin_to_letter(i) for i in Mancala.get_valid_actions(state)]
        action = -1
        while action not in actions:
            action = input("player {}'s next move? ({}): ".format(int(state[-1]), ''.join(actions))).upper()
        return Mancala.letter_to_bin(action)

def play(userGoesFirst=False):
    env = Mancala(render_mode='human', render_unicode=True)
    user = HumanAgent()
    cpu = MCTSAgent(nsamples=3000)

    state, _ = env.reset()
    terminated = False

    while not terminated:
        if (userGoesFirst and state[-1] == 1) or (not userGoesFirst and state[-1] == 2):
            action = user.get_action(state)
        else:
            action = cpu.get_action(state)
        state, _, terminated, _, _ = env.step(action)

if __name__ == "__main__":
    import sys
    try:
        userGoesFirst = int(sys.argv[1]) == 1
    except:
        userGoesFirst = True
    play(userGoesFirst)
