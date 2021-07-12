import pandas as pd
import json
import numpy as np

from scripts.analysis import normalize


class GameResponses(object):
    """ Data structure to write and read cards game task responses from Firestore """

    def __init__(self, randomPick, randomPickColour, reportColour, RTreport, honestyRating, RThonesty,
                 results, catchRating, RTcatch):
        self.randomPick = randomPick
        self.randomPickColour = randomPickColour
        self.reportColour = reportColour
        self.honestyRating = honestyRating
        self.catchRating = catchRating
        self.RThonesty = RThonesty
        self.RTreport = RTreport
        self.RTcatch = RTcatch
        self.results = results

    @staticmethod
    def from_dict(source):
        gresponse = GameResponses(source[u'randomPick'], source[u'randomPickColour'], source[u'reportColour'],
                                  source[u'RTreport'], source[u'honestyRating'], source[u'RThonesty'],
                                  source[u'results'], source[u'catchRating'], source[u'RTcatch'])

        return gresponse

    def to_dict(self):
        dest = {
            u'randomPick': self.randomPick,
            u'randomPickColour': self.randomPickColour,
            u'reportColour': self.reportColour,
            u'RTreport': self.RTreport,
            u'honestyRating': self.honestyRating,
            u'RThonesty': self.RThonesty,
            u'results': self.results,
            u'catchRating': self.catchRating,
            u'RTcatch': self.RTcatch
        }

        return dest


class Demographics(object):
    """ Data structure to read demographics survey responses from Firestore """

    def __init__(self, age, catch, edlev, gender, twin):
        self.age = age
        self.catch = catch
        self.edlev = edlev
        self.gender = gender
        self.twin = twin

    @staticmethod
    def from_dict(source):
        demos = Demographics(source[u'age'], source[u'catch'], source[u'edlev'],
                             source[u'gender'], source[u'twin'])

        return demos

    def to_dict(self):
        dest = {
            u'age': self.age,
            u'catch': self.catch,
            u'edlev': self.edlev,
            u'gender': self.gender,
            u'twin': self.twin
        }

        return dest


def process_trials(trials_json_file: str, n_card_per_trial: int):
    """ preprocess model values to fit from trials setup data; returns trials DataFrame """
    with open(trials_json_file, "r") as read_file:
        trials_source = json.load(read_file)
    trials = pd.DataFrame(trials_source)

    trials['trial'] = trials.index + 1

    trials['n_blue'] = n_card_per_trial - trials['n_red']

    trials['e_v'] = np.where(trials['outcome'] == -1,
                             trials['outcome'] - trials['outcome'] * (trials['n_red'] / n_card_per_trial), \
                             trials['outcome'] - trials['outcome'] * (
                                         n_card_per_trial - trials['n_red']) / n_card_per_trial)
    trials['normed_signed_e_v'] = normalize(trials['e_v'])
    trials['normed_unsigned_e_v'] = normalize(abs(trials['e_v']))

    normalized_signed_colour_count = normalize(trials.signed_n_consec_colour.values)

    trials['normed_signed_colour_count'] = normalized_signed_colour_count
    trials['normed_unsigned_colour_count'] = [(v / 5) for v in trials.n_consec_colour]

    trials['cs_signed_e_v'] = trials['e_v'].cumsum()
    trials['normed_cs_signed_e_v'] = normalize(trials['cs_signed_e_v'])

    trials['cs_unsigned_e_v'] = abs(trials['e_v']).cumsum()
    trials['normed_cs_unsigned_e_v'] = normalize(trials['cs_unsigned_e_v'])

    count_red = abs(trials.outcome[lambda x: x == -1].cumsum())
    count_blue = abs(trials.outcome[lambda x: x == 1].cumsum())

    trials['n_reported_colour_opp'] = count_red.append(count_blue).sort_index()

    track_freq = [1] * len(trials)
    for i, outcome in enumerate(trials['outcome'].values):
        if (i == 0):
            continue
        if (i > 0):
            if (outcome != trials.outcome.values[i - 1]):
                continue
            if (outcome == trials.outcome.values[i - 1]):
                track_freq[i] = track_freq[i - 1] + 1
    trials['n_consec_colour'] = track_freq

    return trials
