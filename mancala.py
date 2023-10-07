import numpy as np

class Mancala:
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, render_unicode=True):
        self.state = []
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_unicode = render_unicode
        self.render_mode = render_mode
        self.observation_space = None # todo
        self.action_space = None # todo

    def initial_game_state(self):
        # encoding:
        #   0  1  2   3  4  5  6  7  8  9  10  11  12 13 14
        #   B1 B2 B3 B4 B5 B6 M1 B7 B8 B9 B10 B11 B12 M2 P
        # where:
        #   Bi = bin i
        #   Mj = mancala for player j
        #   P = player id of whose turn it is
        state = np.zeros(15) # initialize to all zeros
        state[0:6] = 4  # player 1's bins all start with 4 stones
        state[7:13] = 4 # player 2's bins all start with 4 stones
        state[-1] = 1 # player 1 goes first
        return state.tolist()

    def reset(self, seed=None, options=None):
        self.state = self.initial_game_state()
        if self.render_mode == "human":
            self.render(unicode=self.render_unicode)
        return self.state, {}

    @staticmethod
    def is_terminated(state):
        # game is over when one player's bins are all empty
        return sum(state[0:6]) == 0 or sum(state[7:13]) == 0

    @staticmethod
    def next_state(state, action):
        prev_state = [x for x in state]
        state = [x for x in state] # make copy

        # confirm game is not over
        assert not Mancala.is_terminated(state)

        # confirm action is valid given whose turn it is
        if state[-1] == 1:
            assert action in list(range(0,6))
        else:
            assert action in list(range(7,13))
        assert state[action] > 0
        
        # move stones
        remaining = state[action]
        state[action] = 0
        next_bin = action
        while remaining > 0:
            next_bin += 1
            if next_bin > 13: # wrap around
                next_bin = 0
            elif next_bin == 6 and state[-1] == 2:
                # if player 2's turn, don't place in player 1's mancala
                next_bin += 1
            elif next_bin == 13 and state[-1] == 1:
                # if player 1's turn, don't place in player 2's mancala
                next_bin = 0
            state[next_bin] += 1
            remaining -= 1
        next_state = [x for x in state]

        # if last stone was placed in empty bin, current player steals other player's adjacent stones
        if state[next_bin] == 1 and next_bin not in [6,13]:
            # make sure this was our own empty bin
            if (state[-1] == 1 and next_bin < 6) or (state[-1] == 2 and next_bin > 6):
                # steal opponent's stone
                adjacent_bin = 13-(next_bin+1)
                stolen_stones = state[adjacent_bin]
                state[adjacent_bin] = 0
                mancala = 6 if state[-1] == 1 else 13
                state[mancala] += stolen_stones

                # we also get our stone
                state[mancala] += 1
                state[next_bin] = 0

        # if last stone was not placed in mancala, it's now the other player's turn
        if next_bin not in [6,13]:
            state[-1] = (1 if state[-1] == 2 else 2)
        
        assert sum(state[:-1]) == 4*12
        return [int(x) for x in state]

    def step(self, action):
        self.state = Mancala.next_state(self.state, action)
        terminated = Mancala.is_terminated(self.state)
        reward = 0 # todo
        if self.render_mode == "human":
            self.render(action=action, unicode=self.render_unicode)
        return self.state, reward, terminated, False, {}

    @staticmethod
    def get_current_scores(state):
        return sum(state[0:7]), sum(state[7:14])

    @staticmethod
    def get_winner(state):
        # returns winning player
        s1, s2 = Mancala.get_current_scores(state)
        return 1 if s1 > s2 else (2 if s2 > s1 else -1)

    @staticmethod
    def get_valid_actions(state):
        # action encoding is the index of a bin (in state vector) to select

        # figure out what actions are available to that player
        if state[-1] == 1: # player 1 turn
            valid_actions = list(range(0,6))
        else:
            valid_actions = list(range(7,13))
        assert len(valid_actions) == 6

        # can only choose non-empty bins
        valid_actions = [action for action in valid_actions if state[action] > 0]
        return valid_actions

    @staticmethod
    def bin_to_letter(action):
        return 'ABCDEF LKJIHG'[action]

    @staticmethod
    def letter_to_bin(letter):
        return 'ABCDEF LKJIHG'.index(letter)

    @staticmethod
    def count_to_char(count, unicode=False):
        if not unicode:
            return str(int(count))
        if count == 0:
            return ' '
        elif count == 1:
            return '.'
        elif count == 2:
            return ':'
        elif count == 3:
            return '\u2E2A'
        elif count == 4:
            return '\u2E2C'
        elif count == 5:
            return '\u2059'
        elif count == 6:
            return '\u283F'
        elif count == 7:
            return '\u28BF'
        elif count == 8:
            return '\u28FF'
        else:
            return str(int(count))

    def render(self, state=None, action=None, unicode=False):
        #    B12 B11 B10 B9 B8 B7
        # M2                       M1
        #    B1   B2  B3 B4 B5 B6

        state = self.state if state is None else state
        if action is not None:
            print('player chose {}\n'.format(Mancala.bin_to_letter(action)))

        # print game board
        line0 = ' '*5 + ' '.join([Mancala.bin_to_letter(i) for i in range(12,6,-1)])
        line1 = ' '*5 + ' '.join([Mancala.count_to_char(state[i], unicode) for i in range(12,6,-1)])
        line2 = str(int(state[13])) + ' '*20 + str(int(state[6]))
        line3 = ' '*5 + ' '.join([Mancala.count_to_char(state[i], unicode) for i in range(0,6)])
        line4 = ' '*5 + ' '.join([Mancala.bin_to_letter(i) for i in range(0,6)])
        out = ' '*10 + '←\n'
        out += '\n'.join([line0, line1, line2, line3, line4])
        out += '\n' + ' '*10 + '→\n'
        out += '\n\nplayer {} to move'.format(int(state[-1]))

        # print winner
        if Mancala.is_terminated(state):
            out += "\nplayer {} wins: {}".format(Mancala.get_winner(state), Mancala.get_current_scores(state))

        print(out)
        return out
