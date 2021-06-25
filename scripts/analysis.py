import numpy as np
import numexpr

from scripts.specs import Game, Player


class GameResponses(object):
    ''' Data structure to write and read cards game task responses from Firestore '''

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
    ''' Data structure to read demographics survey responses from Firestore '''

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


def reverse_honesty_rating(array):
    return numexpr.evaluate(f'(7 - array)')


def process_trials(trials_json_file):
    #todo
    return


def compute_rsquared(actual_data, predicted_data):
    ''' Compute R^2 with SStotal based on the mean of actual data to fit '''
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    ss_res = np.sum((actual_data - predicted_data) ** 2)
    ss_tot = np.sum((actual_data - actual_data.mean()) ** 2)
    return 1 - np.divide(ss_res, ss_tot)


def fit_signed_suspicion(trials, data, params):
    data = np.array(data)
    ss_tot = np.sum((data - data.mean())**2)
    ss_res = []
    for param in params:
        print(param)
        alpha = param[0]
        prior = param[1]
        estimated = []
        for i in range(len(trials)):
            print(i)
            if i==0:
                estimated.append(trials.normalized_signed_exp_violation.values[i] * alpha + prior)
#                 print(estimated)
            else:
                estimated.append(trials.normalized_signed_exp_violation.values[i] * alpha)
#                 print(estimated)
        ss_res.append(np.sum((data - estimated)**2))
    r2 = 1 - np.divide(ss_res, ss_tot)
    best_idx = np.array(r2).argmax()
    best_params = params[best_idx]
    return best_params, r2


def fit_unsigned_suspicion(trials, data, params):
    data = np.array(data)
    ss_tot = np.sum((data - data.mean())**2)
    ss_res = []
    for param in params:
        print(param)
        alpha = param[0]
        prior = param[1]
        estimated = []
        for i in range(len(trials)):
            print(i)
            if i==0:
                estimated.append(abs(trials.exp_violation).values[i] * alpha + prior)
#                 print(estimated)
            else:
                estimated.append(trials.normalized_signed_exp_violation.values[i] * alpha)
#                 print(estimated)
        ss_res.append(np.sum((data - estimated)**2))
    r2 = 1 - np.divide(ss_res, ss_tot)
    best_idx = np.array(r2).argmax()
    best_params = params[best_idx]
    return best_params, r2


def bulk_fit_rating_delta_from_signed_exp_violation(trials, data, alpha):
    # grid search param value fit
    best_param_SSE = []
    best_param_R2 = []
    best_params = []

    for p, data in enumerate(d_normed_suspicion_ratings):
        print(p)
        data = data[1:]
        SS_tot = np.sum((data - np.mean(data)) ** 2)
        print(SS_tot)
        print(len(data))
        SSres_inter = []
        R2_inter = []
        for i, a in enumerate(alpha):
            print(a)
            print(data)
            SS_res = np.sum(np.subtract(data, a * trials.exp_violation.values[1:]) ** 2)
            SSres_inter.append(SS_res)
            print(SS_res)
            R2 = 1 - SS_res / SS_tot
            R2_inter.append(R2)
            print(R2)
        best_idx = np.array(SSres_inter).argmin()
        best_params.append(alpha[best_idx])
        best_param_SSE.append(np.array(SSres_inter).min())
        best_param_R2.append(R2_inter[best_idx])


def bulk_fit_rating_delta_from_unsigned_exp_violation(trials, data, alpha):
    best_param_SSE = []
    best_param_R2 = []
    best_params = []

    for p, data in enumerate(data):
        print(p)
        data = data[1:]
        data = [(d-data.min())/(data.max()-data.min()) for d in data]
        print(data)
        SS_tot = np.sum((data-np.mean(data))**2)
        SSres_inter = []
        R2_inter = []
        for i, a in enumerate(alpha):
            print(a)
            surprise = abs(trials.exp_violation.values[1:])
            SS_res = np.sum(np.subtract(data, a*surprise)**2)
            SSres_inter.append(SS_res)
            R2 = 1-SS_res/SS_tot
            R2_inter.append(R2)
        best_idx = np.array(SSres_inter).argmin()
        best_params.append(alpha[best_idx])
        best_param_SSE.append(np.array(SSres_inter).min())
        best_param_R2.append(R2_inter[best_idx])