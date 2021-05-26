from specs import Trial, Game

# define trial parameters
n_red = [int(float(n)) for n in input("Specify number of red cards for each consecutive trial, separated by a comma: ").split(',')]

random_trials = input("Do you want to randomize the trials order? (y/n)")
if random_trials == "y":
    random_trials = True
if random_trials == "n":
    random_trials = False
else:
    pass

#todo: add user input for n_trials?

# create trials
trials = []
for n in n_red:
    trials.append(Trial(n))

# create and play game
game = Game(trials, player=None, n_sessions=None, randomize=random_trials)
game.play()
