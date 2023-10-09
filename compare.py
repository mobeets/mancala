from play import play

def sim(player_types, nreps=20, nsamples=[1000,3000]):
	outcomes = []
	print('Players: {}, nreps={}, nsamples={}'.format(player_types, nreps, nsamples))
	for i in range(nreps):
		outcome1 = play(player_types, nsamples, verbose=False, render_mode=None)
		outcome2 = play(player_types[::-1], nsamples, verbose=False, render_mode=None)
		print('{}th outcome: {}, {}'.format(i, outcome1, outcome2))
		outcomes.append((outcome1, outcome2))
	print(outcomes)

if __name__ == '__main__':
	sim(['mcr', 'mcr'])
