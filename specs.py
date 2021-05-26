import random
import csv
import numpy as np

from analysis import normalized_array

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
        # print("You lose 1 point")
        return -1
    elif card_opponent == -1 and card_player == 1:
        # print("You win 1 point")
        return 1
    elif card_opponent == card_player:
        # print("It's a tie: nobody wins or loses a point")
        return 0


def save_log_as_csv(log, filename):
    timestr = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    filename = str(filename) + "-" + timestr + ".csv"
    with open(filename, "w", encoding='utf-8') as f:
        writer = csv.DictWriter(f, log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print("Saved file as", filename)


def softmax(beta, value, values):
    return np.exp(beta * value) / sum(np.exp(beta * values))


class Trial(object):
    """
    Defines a single trial by the proportion of red/blue cards over the total number of cards.
    Outcome represents the value of the other player's reported card.
    """

    n_cards = 5

    def __init__(self, n_red, outcome=None, **kwargs):
        if n_red > self.n_cards:
            raise ValueError("Number of red cards cannot exceed total number of cards (5)")
        self.n_red = n_red  # minimally required setting for a trial
        self.n_blue = self.n_cards - self.n_red
        self.outcome = outcome if outcome is not None else random.choice([-1, 1])
        self.exp_violation = self.outcome - self.expectation()
        self.unsigned_exp_violation = 1 - self.unsigned_expectation()

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

    def unsigned_expectation(self) -> float:
        if self.outcome == -1:
            return self.n_red / self.n_cards
        if self.outcome == 1:
            return self.n_blue / self.n_cards
        else:
            raise ValueError("Opponent outcome has unacceptable value.")

    def selected_card(self):
        return random.choice(self.cards)


class Player(object):
    """
    Player (participant) attributes and possible actions (lie or not lie) based on suspicion model and game rules.
    """

    def __init__(self, alpha, pre_suspicion=0, beta=None, bias=None, **kwargs):
        # todo: estimate alpha and bias from actual playing behaviour - add equations
        self.alpha = alpha
        self.pre_suspicion = pre_suspicion
        self.beta = beta if beta is not None else 1
        self.bias = bias if bias is not None else 0

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

    def __init__(self, trials, player=None, n_sessions=1, randomize=False):
        if randomize:
            random.shuffle(trials)
        self.trials = trials * n_sessions  # expected to be of class Trial
        self.player = player  # expected to be of class Player
        self.player.s0 = self.player.pre_suspicion  # prior suspicion value for first trial
        self.n_sessions = n_sessions
        self.n_trials = len(trials)
        self.sim_log = []
        self.live_log = []
        self.trial_redcards = []
        self.suspicion_values = []
        self.unsigned_expectation_violation = []
        self.softmax_probabilities = []

    def trial_state_values(self, n_red):
        state_idx = [i for i, x in enumerate(self.trial_redcards) if x == n_red]
        # print("number of trials with given n_red:", len(state_idx))
        tmp = np.array(self.suspicion_values)
        return tmp[state_idx]

    def suspicion_to_honesty_rating(self):
        ''' Generate random suspicion ratings in the range [0;1] '''
        normalized_ratings = [0.0, 0.16666666666666663, 0.33333333333333337, 0.5, 0.6666666666666667,
                              0.8333333333333334,
                              1.0]
        normalized_data = normalized_array(self.suspicion_values, self.player.s0)
        return np.array([min(normalized_ratings, key=lambda x: abs(x - sv)) for sv in normalized_data])

    def unsigned_suspicion_to_honesty_rating(self):
        ''' Generate random suspicion ratings in the range [0;1] '''
        normalized_ratings = [0.0, 0.16666666666666663, 0.33333333333333337, 0.5, 0.6666666666666667,
                              0.8333333333333334,
                              1.0]
        return np.array(
            [min(normalized_ratings, key=lambda x: abs(x - sv)) for sv in self.unsigned_expectation_violation])

    def normalize_signed_suspicion(self):
        return normalized_array(self.suspicion_values, self.player.s0)

    def simulate_signed(self, verbose=True, save=False, add_noise=False, set_noise=0.01):
        """ Generate simulated gameplay data with given trials and player settings."""
        if verbose:
            def verboseprint(*args):
                print(*args)
        else:
            verboseprint = lambda *a: None

        if add_noise:
            noise = set_noise * random.uniform(-1, 1)
        else:
            noise = 0

        print("starting game play simulation with signed suspicion")
        print("player attributes:", "prior:", self.player.s0, "alpha:", self.player.alpha)

        for index, t in enumerate(self.trials, start=1):
            print("trial: ", index)
            verboseprint("suspicion_t:", self.player.pre_suspicion)
            verboseprint("# red cards: ", t.n_red)
            self.trial_redcards.append(t.n_red)

            selected_card = t.selected_card()
            verboseprint("randomly picked card for player: ", selected_card)

            # todo: let player randomly "lie" or not - determine based on value from suspicion function in different model?
            # player_selection = random.choice([1, -1])
            verboseprint("opponent card: red") if t.outcome == -1 else verboseprint("opponent card: blue")

            # trial outcome
            # todo: assumes player always reports honestly, i.e. the randomly selected_card; change probabilistically?
            t.player_reward = reward(t.outcome, selected_card)
            t.opponent_reward = reward(selected_card, t.outcome)

            # player suspicion according to suspicion formula, resembles Q update rule in Rescorla-Wagner learning
            old_suspicion = self.player.pre_suspicion
            new_suspicion = old_suspicion + self.player.alpha * t.exp_violation + noise
            delta_suspicion = abs(new_suspicion - old_suspicion)
            self.suspicion_values.append(new_suspicion)
            self.player.update_suspicion(new_suspicion)
            verboseprint("change in suspicion: ", delta_suspicion)
            verboseprint("new player suspicion: ", new_suspicion, "\n")

            # softmax probability for reverse-coded honesty rating
            probability = softmax(self.player.beta, new_suspicion, self.trial_state_values(t.n_red))
            self.softmax_probabilities.append(probability)
            verboseprint("probability of suspicion rating: ", probability)

            # log game
            self.sim_log.append({"trial": index,
                                 "n_red": t.n_red,
                                 "n_blue": t.n_blue,
                                 "partner_card_colour": t.outcome,
                                 "partner_card_colour_expectation": t.expectation(),
                                 "expectation_violation": t.exp_violation,
                                 "suspicion_tmin1": old_suspicion,
                                 "suspicion_t": new_suspicion,
                                 "absolute_suspicion_change": delta_suspicion,
                                 # "random_pick": selected_card,
                                 # "played_card": selected_card,  # todo: change to player_selection if allowing lies
                                 # "player_reward": t.player_reward,
                                 # "opponent_reward": t.opponent_reward,
                                 # "softmax_probability": probability
                                 })
        print("end of simulated game")
        if save:
            filename = input("save as: ")
            save_log_as_csv(self.sim_log, filename)

    def simulate_unsigned(self, verbose=True, save=False, add_noise=False, set_noise=0.01):
        """ Generate simulated gameplay data with given trials and player settings."""
        if verbose:
            def verboseprint(*args):
                print(*args)
        else:
            verboseprint = lambda *a: None

        if add_noise:
            noise = set_noise * random.uniform(-1, 1)
        else:
            noise = 0

        print("starting game play simulation with unsigned suspicion")
        print("player attributes:", "prior:", self.player.s0, "alpha:", self.player.alpha)

        for index, t in enumerate(self.trials, start=1):
            print("trial: ", index)
            verboseprint("suspicion_t:", self.player.pre_suspicion)
            verboseprint("# red cards: ", t.n_red)
            self.trial_redcards.append(t.n_red)

            selected_card = t.selected_card()
            verboseprint("randomly picked card for player: ", selected_card)

            # todo: let player randomly "lie" or not - determine based on value from suspicion function in different model?
            # player_selection = random.choice([1, -1])
            verboseprint("opponent card: red") if t.outcome == -1 else verboseprint("opponent card: blue")

            # trial outcome
            # todo: assumes player always reports honestly, i.e. the randomly selected_card; change probabilistically?
            t.player_reward = reward(t.outcome, selected_card)
            t.opponent_reward = reward(selected_card, t.outcome)

            # player suspicion according to suspicion formula, resembles Q update rule in Rescorla-Wagner learning
            old_suspicion = self.player.pre_suspicion
            new_suspicion = self.player.s0 + t.unsigned_exp_violation * self.player.alpha + noise
            self.unsigned_expectation_violation.append(new_suspicion)
            delta_suspicion = abs(new_suspicion - old_suspicion)
            verboseprint("change in suspicion: ", delta_suspicion)
            verboseprint("new player suspicion: ", new_suspicion, "\n")

            # log game
            self.sim_log.append({"trial": index,
                                 "n_red": t.n_red,
                                 "n_blue": t.n_blue,
                                 "opponent_card": t.outcome,
                                 "expectation": t.expectation(),
                                 "unsigned_exp_violation": t.unsigned_exp_violation,
                                 "suspicion_tmin1": old_suspicion,
                                 "suspicion_t": new_suspicion,
                                 "delta_suspicion": delta_suspicion,
                                 "random_pick": selected_card,
                                 "player_reward": t.player_reward,
                                 "opponent_reward": t.opponent_reward})
        print("end of simulated game")
        if save:
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
                "How honest do you think the other player is? (0 = not at all, 6 = completely)")

            self.live_log.append({
                "trial": index,
                "n_red": t.n_red,
                "n_blue": t.n_blue,
                "opponent_card": t.outcome,
                "expectation": t.expectation(),
                "exp_violation": t.exp_violation,
                "player_pick": selected_card,
                "played_card": player_selection,
                "reward_t": t.reward,
                "RT_report": response_time,
                "honesty_rating": rating})
            print("=" * 100)
        print("You have reached the end of the game.")
        save_input = input("Would you like to save your results? (y/n)")
        if save_input == "y":
            filename = input("Save as: ")
            save_log_as_csv(self.live_log, filename)


