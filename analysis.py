import numpy as np


def normalize(x):
    # todo: adjust denominator value according to which free parameters are being estimated
    return x/5.4  # NOTE: this is based on game trials order with Player alpha=1 and no other free parameters

def normalized_array(array, s0):
    tmp = [x+abs(min(array)) for x in array]
    return np.array([normalize(x+s0) for x in tmp])

def compute_rsquared(actual_data, predicted_data):
    ''' Compute R^2 with SStotal based on the mean of actual data to fit '''
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    ss_res = sum(actual_data - predicted_data) ** 2
    ss_tot = sum(actual_data - actual_data.mean()) ** 2
    r2 = 1 - ss_res / ss_tot
    return r2
