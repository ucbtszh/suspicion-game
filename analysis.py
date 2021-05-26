import numpy as np


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


def normalize(x):
    # todo: adjust denominator value according to which free parameters are being estimated
    return x / 5.4  # NOTE: this is based on game trials order with Player alpha=1 and no other free parameters


def normalized_array(array, s0):
    tmp = [x + abs(min(array)) for x in array]
    return np.array([normalize(x + s0) for x in tmp])


def compute_rsquared(actual_data, predicted_data):
    ''' Compute R^2 with SStotal based on the mean of actual data to fit '''
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    ss_res = np.sum((actual_data - predicted_data) ** 2)
    ss_tot = np.sum((actual_data - actual_data.mean()) ** 2)
    return 1 - np.divide(ss_res, ss_tot)
