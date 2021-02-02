import random
import csv

from time import time
from datetime import datetime

random.seed(42)

# helper functions
def delta(x):
    if len(x) > 1:
        return [y - x for x, y in zip(x, x[1:])]
    return "The game first needs to be played before temporal differences can be computed: Game.play() or Game.play_simulate()"

def reward(card_opponent, card_player):
    if card_opponent == 1 and card_player == -1:
        print("You lose 1 point")
        return -1
    elif card_opponent == -1 and card_player == 1:
        print("You win 1 point")
        return 1
    elif card_opponent == 1 and card_player == 1:
        print("You and your opponent lose 1 point")
        return -1
    elif card_opponent == -1 and card_player == -1:
        print("You and your opponent win 1 point")
        return 1

def save_log_as_csv(log, filename):
    timestr = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    filename = str(filename) + "-" + timestr + ".csv"
    with open(filename, "w", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(log)
    print("Saved file as", filename)


class Trial(object):
    """
    Defines a single trial by the proportion of red/blue cards over the total number of cards.
    """

    n_cards = 5

    def __init__(self, n_red, outcome=None, **kwargs):
        if n_red > self.n_cards:
            raise ValueError("Number of red cards cannot exceed total number of cards (5)")
        self.n_red = n_red  # minimally required setting for a trial
        self.n_blue = self.n_cards - self.n_red
        self.outcome = outcome if outcome is not None else random.choice([-1, 1])
        self.exp_violation = self.outcome - self.expectation()

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

    def __init__(self, alpha, bias=None, pre_suspicion=None, beta=None, **kwargs):
        # todo: estimate alpha and bias from actual playing behaviour - add equations
        self.alpha = alpha
        self.bias = bias if bias is not None else 0
        self.pre_suspicion = pre_suspicion if pre_suspicion is not None else 0 # initialize first trial pre_suspicion=0
        self.beta = beta if beta is not None else 1

        # add any further specified parameters
        for key, value in kwargs:
            self.key = value

    def update_suspicion(self, value):
        self.pre_suspicion = value

    def update_bias(self, value):
        self.bias = value


class Game(object):
    """
    Lets Player interact with given Trial(s) according to acquired suspicion level.
    """

    def __init__(self, trials, player=None, n_trials=None, randomize=False):
        self.trials = trials  # expected to be of class Trial
        self.player = player # expected to be of class Player
        self.n_trials = n_trials if n_trials is not None \
            else len(trials)  # todo: update self.trials if n_trials < or > len(trials)

        if randomize:
            random.shuffle(trials)

        self.sim_log = ["trial",
                        "n_red",
                        "n_blue",
                        "opponent_card",
                        "expectation",
                        "exp_violation",
                        "suspicion_tmin1",
                        "learning_rate",
                        "bias",
                        "delta_suspicion",
                        "suspicion_t",
                        "player_pick",
                        "played_card",
                        "reward_t"]

        self.live_log = ["trial",
                        "n_red",
                        "n_blue",
                        "opponent_card",
                        "expectation",
                        "exp_violation",
                        "player_pick",
                        "played_card",
                        "reward_t",
                         "RT",
                         "rating"]

    def simulate_play(self, verbose=True):
        """ Simulate gameplay with given trials and player settings."""
        if verbose:
            def verboseprint(*args):
                print(*args)
        else:
            verboseprint = lambda *a: None

        print("starting game play simulation")
        verboseprint("player attributes: bias: ", self.player.bias, "alpha: ", self.player.alpha)

        for index, t in enumerate(self.trials, start=1):
            print("trial: ", index)
            verboseprint("suspicion_t:", self.player.pre_suspicion)
            verboseprint("# red cards: ", t.n_red)

            selected_card = t.selected_card()
            verboseprint("randomly picked card for player: ", selected_card)

            # todo: for now, let player randomly "lie" or not - determine based on value-suspicion function
            player_selection = random.choice([1, -1])
            verboseprint("opponent card: red") if t.outcome == -1 else verboseprint("opponent card: blue")

            t.reward = reward(t.outcome, player_selection)

            # player suspicion according to suspicion formula
            new_suspicion = self.player.pre_suspicion + self.player.bias + self.player.alpha * t.exp_violation
            delta_suspicion = self.player.pre_suspicion - new_suspicion
            verboseprint("change in suspicion: ", delta_suspicion)
            verboseprint("new player suspicion: ", new_suspicion, "\n")

            # log game
            self.sim_log.append([index,
                                 t.n_red,
                                 t.n_blue,
                                 t.outcome,
                                 t.expectation(),
                                 t.exp_violation,
                                 self.player.pre_suspicion,
                                 self.player.alpha,
                                 self.player.bias,
                                 delta_suspicion,
                                 new_suspicion,
                                 selected_card,
                                 player_selection,
                                 t.reward])
            self.player.update_suspicion(new_suspicion)
        print("end of simulated game")
        save_input = input("save log? (y/n)")
        if save_input == "y":
            filename = input("save as: ")
            save_log_as_csv(self.sim_log, filename)

    def play(self):
        """ Play game using command line interface. """
        print("=" * 100)
        print("Welcome to the Lucky Card game! \n"
              "In each round, you and your opponent will see the identical set of cards. \n"
              "The computer randomly selects a card for you and your opponent. \n"
              "You need to decide whether to play this card or to play a card with a different colour. \n"
              "\n"
              "Good luck.")
        print("=" * 100)
        print("=" * 100)
        for index, t in enumerate(self.trials, start=1):
            selected_card = t.selected_card()

            print("Round", index)
            print("There are", t.n_red, "red cards and", t.n_blue, "blue cards")
            print("Your randomly selected playing card is red") if selected_card == -1 else print(
                "Your randomly selected playing card is blue")

            # catch plyaer RT and store decision
            starttime = time()
            player_selection = input("Which card do you wish to play? (red/blue)")
            response_time = time() - starttime

            print("You selected red") if player_selection == "red" else print("You selected blue")
            print("Your opponent selected: red card") if t.outcome == -1 else print("Your opponent selected: blue card")

            if player_selection == "red":
                player_selection = -1
            else:
                player_selection = 1
            t.reward = reward(t.outcome, player_selection)

            rating = input(
                "On a scale from 0-10, how confident are you in that your opponent played honestly? (0 = not at all, 10 = completely)")

            self.live_log.append([index,
                                  t.n_red,
                                  t.n_blue,
                                  t.outcome,
                                  t.expectation(),
                                  t.exp_violation,
                                  selected_card,
                                  player_selection,
                                  t.reward,
                                  response_time,
                                  rating])
            print("=" * 100)
        print("You have reached the end of the game.")
        save_input = input("Would you like to save your results? (y/n)")
        if save_input == "y":
            filename = input("Save as: ")
            save_log_as_csv(self.live_log, filename)
