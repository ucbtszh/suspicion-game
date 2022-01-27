import pandas as pd
import json
import numpy as np

from scripts.analysis import normalize


class GameResponses(object):
    """ Data structure to write and read cards game task responses from Firestore """

    def __init__(self, randomPick, randomPickColour, reportColour, RTreport, honestyRating, RThonesty,
                 results, catchRating, RTcatch, nred, ppLied, otherReport):
        self.randomPick = randomPick
        self.randomPickColour = randomPickColour
        self.reportColour = reportColour
        self.honestyRating = honestyRating
        self.catchRating = catchRating
        self.RThonesty = RThonesty
        self.RTreport = RTreport
        self.RTcatch = RTcatch
        self.results = results
        self.nRed = nred
        self.ppLied = ppLied
        self.otherReport = otherReport

    @staticmethod
    def from_dict(source):
        gresponse = GameResponses(source[u'randomPick'], source[u'randomPickColour'], source[u'reportColour'],
                                  source[u'RTreport'], source[u'honestyRating'], source[u'RThonesty'],
                                  source[u'results'], source[u'catchRating'], source[u'RTcatch'], source[u'nRed'],
                                  source[u'ppLied'], source[u'outcome'])

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
            u'RTcatch': self.RTcatch,
            u'nRed': self.nRed,
            u'ppLied': self.ppLied,
            u'otherReport': self.otherReport
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



def process_trials_from_df(df_trials, n_card_per_trial: int):
    """ preprocess model values to fit from trials setup data; assumes trials to be of type pandas DataFrame """
    trials = df_trials

    trials['trial'] = trials.index + 1
    trials['n_blue'] = n_card_per_trial - trials['n_red']

    # within-trials expectation violation as function of reported card colour likelihood
    trials['e_v'] = np.where(trials['outcome'] == -1,
                             trials['outcome'] - trials['outcome'] * (trials['n_red'] / n_card_per_trial), \
                             trials['outcome'] - trials['outcome'] * (
                                     n_card_per_trial - trials['n_red']) / n_card_per_trial)
    trials['normed_signed_e_v'] = normalize(trials['e_v'])
    trials['normed_unsigned_e_v'] = normalize(abs(trials['e_v']))

    # # across-trials 'global' count of consecutive card colour reports
    # count_red = abs(trials.outcome[lambda x: x == -1].cumsum())
    # count_blue = abs(trials.outcome[lambda x: x == 1].cumsum())
    # trials['n_reported_colour_opp'] = count_red.append(count_blue).sort_index()
    #
    # track_freq = [1] * len(trials)
    # for i, outcome in enumerate(trials['outcome'].values):
    #     if (i == 0):
    #         continue
    #     if (i > 0):
    #         if (outcome != trials.outcome.values[i - 1]):
    #             continue
    #         if (outcome == trials.outcome.values[i - 1]):
    #             track_freq[i] = track_freq[i - 1] + 1
    #
    # trials['n_consec_colour'] = track_freq
    # normalized_signed_colour_count = normalize(trials['n_consec_colour'] * trials['outcome'])
    #
    # trials['normed_signed_colour_count'] = normalized_signed_colour_count
    # trials['normed_unsigned_colour_count'] = normalize(trials['n_consec_colour'])

    return trials


def global_col_count(trials):
    # with trial-to-trial update rule, reuslts in cumulcative sum of expectation violations:
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
    normalized_signed_colour_count = normalize(trials['n_consec_colour'] * trials['outcome'])

    trials['normed_signed_colour_count'] = normalized_signed_colour_count
    trials['normed_unsigned_colour_count'] = normalize(trials['n_consec_colour'])

    return trials


def process_trials_from_file(trials_json_file: str, n_card_per_trial: int):
    """ preprocess model values to fit from trials setup data; returns trials DataFrame """
    with open(trials_json_file, "r") as read_file:
        trials_source = json.load(read_file)
    trials = pd.DataFrame(trials_source)

    trials['trial'] = trials.index + 1
    trials['n_blue'] = n_card_per_trial - trials['n_red']

    # within-trials expectation violation as function of reported card colour likelihood
    trials['e_v'] = np.where(trials['outcome'] == -1,
                             trials['outcome'] - trials['outcome'] * (trials['n_red'] / n_card_per_trial), \
                             trials['outcome'] - trials['outcome'] * (
                                     n_card_per_trial - trials['n_red']) / n_card_per_trial)
    trials['normed_signed_e_v'] = normalize(trials['e_v'])
    trials['normed_unsigned_e_v'] = normalize(abs(trials['e_v']))

    # with trial-to-trial update rule, reuslts in cumulcative sum of expectation violations:
    trials['cs_signed_e_v'] = trials['e_v'].cumsum()
    trials['normed_cs_signed_e_v'] = normalize(trials['cs_signed_e_v'])

    trials['cs_unsigned_e_v'] = abs(trials['e_v']).cumsum()
    trials['normed_cs_unsigned_e_v'] = normalize(trials['cs_unsigned_e_v'])

    # across-trials 'global' count of consecutive card colour reports
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
    normalized_signed_colour_count = normalize(trials['n_consec_colour'] * trials['outcome'])

    trials['normed_signed_colour_count'] = normalized_signed_colour_count
    trials['normed_unsigned_colour_count'] = normalize(trials['n_consec_colour'])

    return trials


def same_cond_set1_trials(uuid, condition, as31, bs11, bs02):
    if condition[uuid] == "11":
        return pd.concat([bs11, as31, bs02]).reset_index()
    elif condition[uuid] == "12":
        return pd.concat([bs11, bs02, as31]).reset_index()
    elif condition[uuid] == "21":
        return pd.concat([as31, bs11, bs02]).reset_index()
    elif condition[uuid] == "22":
        return pd.concat([as31, bs02, bs11]).reset_index()
    elif condition[uuid] == "31":
        return pd.concat([bs02, bs11, as31]).reset_index()
    elif condition[uuid] == "32":
        return pd.concat([bs02, as31, bs11]).reset_index()
    else:
        print('no applicable subject condition found')
