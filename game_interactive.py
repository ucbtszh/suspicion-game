import random
from specs import Trial, Game

# which trial params to use? default / user specified?
n_red = list(input("Specify number of red cards for each consecutive trial, separated by comma: "))
random_trials = input("Do you want to randomize the trials order? (y/n)")
if random_trials == "y":
    random_trials = True
if random_trials == "n":
    random_trials = False
else:
    pass

outcome = []
for i in range(len(n_red)):
    outcome.append(random.choice([1, -1]))
#todo: add user input for n_trials?

# create trials
trial_params = list(zip(n_red, outcome))
trials = []
for params in trial_params:
    trials.append(Trial(params))

# create and play game
game = Game(trials, random_trials)
game.play()
