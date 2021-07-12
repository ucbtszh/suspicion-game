import numpy as np
import numexpr

from functools import partial
from skopt import gp_minimize
from sklearn.metrics import mean_squared_error


def normalize(data):
    return (data + abs(min(data)))/(abs(min(data)) + abs(max(data)))


def reverse_honesty_rating(array):
    return numexpr.evaluate(f'(6 - array)')


def normalize_7point_ratings(data):
    return numexpr.evaluate('(1 - data)')


def compute_rsquared(actual_data, predicted_data):
    ''' Compute R^2 with SStotal based on the mean of actual data to fit '''
    actual_data = np.array(actual_data)
    predicted_data = np.array(predicted_data)
    ss_res = np.sum((actual_data - predicted_data) ** 2)
    ss_tot = np.sum((actual_data - actual_data.mean()) ** 2)
    return 1 - np.divide(ss_res, ss_tot)


def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def gridsearch_single_model(responses, params, trials, stat: str):
    best_params = []
    best_r2 = []
    for i, r in enumerate(responses):
        print('response subject', i)
        ss_tot = np.sum((r-np.mean(r))**2)
        print('ss totals', ss_tot)
        r2 = []
        for param in params:
            pred = param[1] + param[0] * trials[stat]
            ss_res = np.sum((r-pred)**2)
            r2.append(1 - np.divide(ss_res, ss_tot))
        r2 = np.array(r2)
        best_idx = r2.argmax()
        print('best r2 index', best_idx)
        best_params.append(params[best_idx])
        best_r2.append(r2.max())
    return best_params, best_r2


def objective_single_model(params, response, trials, stat: str):
    pred = params[0] + params[1] * trials[stat]
    return np.sum((response-pred)**2)


def skopt_fit_single_model(responses, trials, param_search_space):
    for i, response in enumerate(responses):
        ss_tot = np.sum((response - np.mean(response)) ** 2)
        gp_result = gp_minimize(
            partial(objective_single_model, response=response, trials=trials, stat='normed_cs_signed_e_v'),
            param_search_space, random_state=42)
        optimal_ss_res = gp_result.fun
        print("Subject", i)
        print("Best parameter estimates: prior =", (gp_result.x[0], "alpha =", gp_result.x[1]))
        print("R2:", 1 - np.divide(optimal_ss_res, ss_tot))
        pred = gp_result.x[0] + gp_result.x[1] * trials['normed_cs_signed_e_v']
        mse = mean_squared_error(response, pred)
        bic = calculate_bic(len(response), mse, len(param_search_space))
        aic = calculate_aic(len(response), mse, len(param_search_space))
        print("BIC:", bic)
        print("AIC:", aic)
        print("=" * 100)


def objective_weighted(params, response, trials):
    pred = params[0] + params[1] * trials['normed_signed_e_v'] + params[2] * trials['normed_signed_colour_count']
    ss_res = np.sum((response-pred)**2)
    return ss_res


def skopt_fit_weighted(responses, trials, param_search_space):
    for i, response in enumerate(responses):
        ss_tot = np.sum((response - np.mean(response)) ** 2)
        gp_result = gp_minimize(partial(objective_weighted, response=response, trials=trials), param_search_space,
                                random_state=42)
        optimal_ss_res = gp_result.fun
        print("Subject", i)
        print("Best parameter estimates: prior =",
              (gp_result.x[0], "alpha 1 =", gp_result.x[1], "alpha 2 =", gp_result.x[2]))
        print("R2:", 1 - np.divide(optimal_ss_res, ss_tot))
        pred = gp_result.x[0] + gp_result.x[1] * trials['normed_signed_e_v'] + gp_result.x[2] * trials[
            'normed_signed_colour_count']
        mse = mean_squared_error(response, pred)
        bic = calculate_bic(len(response), mse, len(param_search_space))
        aic = calculate_aic(len(response), mse, len(param_search_space))
        print("BIC:", bic)
        print("AIC:", aic)
        print("=" * 100)