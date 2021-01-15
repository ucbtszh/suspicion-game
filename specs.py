import random
import csv
from time import time

random.seed(42)

def suspicion(pre_suspicion, alpha, baseline, outcome, expectation) -> float:
    """
    Returns suspicion level according to hypothesized computational behaviour model.
    """
    return pre_suspicion + alpha * (outcome - expectation) + baseline

class Trial(object):
    """
    One game play trial is defined by the proportion of red/blue cards over the total number of cards and the opponent's shown outcome.
    """

    n_cards = 5

    def __init__(self, trial_params, **kwargs):
        # minimum requirement for a trial are number of red cards and opponent outcome, expected as trial_params
        self.n_red = trial_params[0]
        self.outcome = trial_params[1]
        self.n_blue = self.n_cards - self.n_red

        # add any further specified parameters
        for key, value in kwargs:
            self.key = value

        # create all playing cards: blue card value = 1, red card value = -1
        self.cards = []
        self.cards.extend([-1] * self.n_red)
        self.cards.extend([1] * self.n_blue)

    def expectation(self) -> float:
        if self.outcome == -1:
            return self.outcome * self.n_red / self.n_cards
        if self.outcome == 1:
            return self.outcome * self.n_blue / self.n_cards
        else:
            raise ValueError("Opponent outcome has unacceptable value.")

    def selected_card(self):
        return random.choice(self.cards)


class Player(object):
    """
    Player (participant) attributes and possible actions (lie or not lie) based on suspicion model and game rules.
    """

    def __init__(self, player_params, pre_suspicion=None, **kwargs):
        # todo: estimate alpha and baseline from actual playing behaviour - add equations
        self.alpha = player_params[0]
        self.baseline = player_params[1]

        # first trial has no pre_suspicion
        self.pre_suspicion = pre_suspicion if pre_suspicion is not None else 0

    def update_suspicion(self, value):
        self.pre_suspicion = value


class Game(object):
    """
    Lets Player interact with given Trial(s) according to acquired suspicion level.
    """

    def __init__(self, trials, player=None, n_trials=None, randomize=False):
        self.trials = trials  # expected to be of class Trial
        if player is not None:
            self.player = player  # expected to be of class Player
            self.player_suspicion = [self.player.pre_suspicion]
        self.n_trials = n_trials if n_trials is not None else len(trials) # todo: update self.trials if n_trials < or > len(trials)
        if randomize:
            random.shuffle(trials)
        self.player_log = []

    def play_simulate(self, verbose=True):
        """ Simulate gameplay with given trial and player settings. """
        if verbose:
            def verboseprint(*args):
                print(*args)
        else:
            verboseprint = lambda *a: None

        for index, t in enumerate(self.trials, start=1):
            print("trial: ", index)
            verboseprint("player attributes: baseline: ", self.player.baseline, "alpha: ", self.player.alpha,
                         "pre-suspicion: ", self.player.pre_suspicion)
            verboseprint("# red cards: ", t.n_red)

            # define playing options
            selected_card = t.selected_card()
            verboseprint("card selected for player: ", selected_card)
            lie_card = 1 if selected_card == -1 else -1

            # player decision based on existing suspicion
            if self.player.alpha * (t.outcome - t.expectation()) > self.player.baseline + self.player.pre_suspicion:
                player_selection = lie_card
                print("player lies: ", player_selection)
            else:
                player_selection = selected_card
                print("player plays selected card: ", player_selection)

            verboseprint("opponent card: red") if t.outcome == -1 else verboseprint("opponent card: blue")

            if player_selection > t.outcome:
                print("player wins \n")
            elif player_selection < t.outcome:
                print("player loses \n")
            else:
                print("it's a tie \n")

            # update player suspicion and save value
            new_suspicion = suspicion(self.player.pre_suspicion, self.player.alpha, self.player.baseline, t.outcome,
                                      t.expectation())
            self.player_suspicion.append(new_suspicion)
            self.player.update_suspicion(new_suspicion)
            self.player_log.append(
                [index, t.n_red, t.n_blue, t.outcome, t.expectation(), selected_card, player_selection, new_suspicion])
            verboseprint("new player suspicion: ", new_suspicion, "\n")

    def play(self):
        """ Play game using command line interface. """
        for index, t in enumerate(self.trials, start=1):
            print("You are in trial", index)
            print("There are", t.n_red, "red cards and", t.n_blue, "blue cards")
            selected_card = t.selected_card()
            print("Your suggested playing card is red") if selected_card == -1 else print("Your suggested playing card is red")

            # catch plyaer RT and store decision
            starttime = time()
            player_selection = input("Which colour do you wish to play? (red/blue)")
            response_time = time() - starttime

            print("You selected red") if player_selection == "red" else print("You selected blue")
            print("Your opponent selected: red card") if t.outcome == -1 else print("Your opponent selected: blue card")

            if player_selection == "blue" and t.outcome == -1:
                print("player wins \n")
            elif player_selection == "red" and t.outcome == 1:
                print("player loses \n")
            else:
                print("it's a tie \n")

            q_suspect = input("On a scale from 0-10, how confident are you in that your opponent played honestly? (0 = not at all, 10 = completely)" )

            self.player_log.append(
                [index, t.n_red, t.n_blue, t.outcome, t.expectation(), selected_card, player_selection, response_time,
                 q_suspect])
            print("="*100)
        print("You have reached the end of the game.")

        save_input = input("Would you like to save your results? (y/n)")
        if save_input == "y":
            filename = input("Save as: ") + ".csv"
            with open(filename, "w") as f:
                writer = csv.writer(f)
                writer.writerows(self.player_log)

    def delta_suspicion(self):
        if len(self.player_suspicion) > 1:
            return [y - x for x, y in zip(self.player_suspicion, self.player_suspicion[1:])]
        return "The game first needs to be played before suspicion can be developed: Game.play() or Game.play_simulate()"
