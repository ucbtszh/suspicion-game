import random

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

    def __init__(self, trial_params):
        # blue card value = 1, red card value = -1
        self.n_red = trial_params[0]
        self.outcome = trial_params[1]
        self.n_blue = self.n_cards - self.n_red

        # create all playing cards
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
        random.seed(42)
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

    def __init__(self, trials, player, n_trials=None):
        self.trials = trials  # expected to be of class Trial
        self.player = player  # expected to be of class Player
        self.player_suspicion = [self.player.pre_suspicion]
        self.n_trials = n_trials if n_trials is not None else len(
            trials)  # todo: update self.trials if n_trials < or > len(trials)

    def play(self):
        for index, t in enumerate(self.trials, start=1):
            print("trial: ", index)
            print("player attributes: baseline: ", self.player.baseline, "alpha: ", self.player.alpha, "pre-suspicion: ", self.player.pre_suspicion)
            print("# red cards: ", t.n_red)

            # define playing options
            selected_card = t.selected_card()
            print("card selected for player: ", selected_card)
            lie_card = 1 if selected_card == -1 else -1

            # player decision based on existing suspicion
            if self.player.alpha * (t.outcome - t.expectation()) > self.player.baseline + self.player.pre_suspicion:
                print("player lies: ", lie_card)
            else:
                print("player plays selected card: ", selected_card)

            print("opponent card: red") if t.outcome == -1 else print("opponent card: blue")

            # update player suspicion and save value
            new_suspicion = suspicion(self.player.pre_suspicion, self.player.alpha, self.player.baseline, t.outcome,
                                      t.expectation())
            self.player_suspicion.append(new_suspicion)
            self.player.update_suspicion(new_suspicion)
            print("new player suspicion: ", new_suspicion, "\n")

    def delta_suspicion(self):
        if len(self.player_suspicion) > 1:
            return [y - x for x, y in zip(self.player_suspicion, self.player_suspicion[1:])]
        return "The game first needs to be played before suspicion can be developed: Game.play()"
        