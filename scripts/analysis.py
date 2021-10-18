import numpy as np
import numexpr

from functools import partial
from skopt import gp_minimize
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, mean_squared_error


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


def skopt_fit_single_model_single_response(response, trials, param_search_space, stat: str):
    ss_tot = np.sum((response - np.mean(response)) ** 2)
    
    gp_result = gp_minimize(
        partial(objective_single_model, response=response, trials=trials, stat=stat),
        param_search_space, random_state=42)
    optimal_ss_res = gp_result.fun
    
    r2 =  1 - np.divide(optimal_ss_res, ss_tot)
    
    print("Best parameter estimates: prior =", (gp_result.x[0], "alpha =", gp_result.x[1]))
    print("R2:", r2)
    
    pred = gp_result.x[0] + gp_result.x[1] * trials[stat]
    
    mse = mean_squared_error(response, pred)
    bic = calculate_bic(len(response), mse, len(param_search_space))
    aic = calculate_aic(len(response), mse, len(param_search_space))
    
    print(len(response), mse, len(param_search_space))
    
    print("BIC:", bic)
    print("AIC:", aic)
    print("=" * 100)
    
    return {"ss_total": ss_tot, "est_prior": gp_result.x[0], "est_alpha": gp_result.x[1], "mean_squared_error": mse, "R2": r2, "BIC": bic, "AIC": aic}
    

def objective_weighted(params, response, trials, stat1: str, stat2: str):
    pred = params[0] + params[1] * trials[stat1] + params[2] * trials[stat2]
    ss_res = np.sum((response-pred)**2)
    return ss_res


def skopt_fit_weighted_model_single_response(response, trials, param_search_space, stat1: str, stat2: str):
    ss_tot = np.sum((response - np.mean(response)) ** 2)
    gp_result = gp_minimize(partial(objective_weighted, response=response, trials=trials, stat1=stat1, stat2=stat2),
                            param_search_space, random_state=42)
    optimal_ss_res = gp_result.fun
    
    r2 = 1 - np.divide(optimal_ss_res, ss_tot)

    print("Best parameter estimates: prior =", gp_result.x[0], "alpha 1 =", gp_result.x[1], "alpha 2 =", gp_result.x[2])
    print("R2:", r2)

    pred = gp_result.x[0] + gp_result.x[1] * trials[stat1] + gp_result.x[2] * trials[stat2]

    mse = mean_squared_error(response, pred)
    bic = calculate_bic(len(response), mse, len(param_search_space))
    aic = calculate_aic(len(response), mse, len(param_search_space))

    print("BIC:", bic)
    print("AIC:", aic)
    print("=" * 100)
    
    return {"ss_total": ss_tot, "est_prior": gp_result.x[0], "est_alpha1": gp_result.x[1], "est_alpha2": gp_result.x[2], "mean_squared_error": mse, "R2": r2, "BIC": bic, "AIC": aic}


def reset_n_blue_per_trial():
    return {1: {'blue': 0, 'red': 0},
                    2: {'blue': 0, 'red': 0},
                    3: {'blue': 0, 'red': 0},
                    4: {'blue': 0, 'red': 0},
                    5: {'blue': 0, 'red': 0},
                    6: {'blue': 0, 'red': 0}}


def factorial(n):
    if n < 2:
        return 1
    else:
        return n * factorial(n-1)


def check_lie_prob_signed(n_blue_per_trial, n_cards, n_red, outcome):
    p_blue_trial = (n_cards - n_red) / n_cards

    if outcome == 1:
        n_blue_per_trial[n_red]['blue'] += 1
    elif outcome == -1:
        n_blue_per_trial[n_red]['red'] += 1
    else:
        raise ValueError("Unknown report colour value input")

    n_trials = n_blue_per_trial[n_red]['blue'] + n_blue_per_trial[n_red]['red']
    print("n trials with setup", n_trials)

    if n_trials == 1:
        if outcome == 1:
            print("p blue in trial", p_blue_trial)
            return 1 - p_blue_trial
        else:
            return 1  # because we assume people only lie rationally, if outcome is -1 (red) it should not raise suspicion (very high likelihood)

    n_trials_blue = n_blue_per_trial[n_red]['blue']

    n_combinations = factorial(n_trials)/(factorial(n_trials_blue) * factorial(n_trials - n_trials_blue))

    p_red_trial = n_red / n_cards

    if outcome == -1:
        return 1
    return (p_blue_trial ** n_trials_blue) * (p_red_trial ** (n_trials - n_trials_blue)) * n_combinations


def check_lie_prob(n_blue_per_trial, n_cards, n_red, outcome):
    p_blue_trial = (n_cards - n_red) / n_cards

    if outcome == 1:
        n_blue_per_trial[n_red]['blue'] += 1
    elif outcome == -1:
        n_blue_per_trial[n_red]['red'] += 1
    else:
        raise ValueError("Unknown report colour value input")

    n_trials = n_blue_per_trial[n_red]['blue'] + n_blue_per_trial[n_red]['red']
    print("n trials with setup", n_trials)

    if n_trials == 1:
        if outcome == 1:
            print("p blue in trial", p_blue_trial)
            return 1 - p_blue_trial
        else:
            return p_blue_trial

    if (n_blue_per_trial[n_red]['blue'] == 0) | (n_blue_per_trial[n_red]['red'] == 0):
        n_combinations = 1
    #         print("n combi 1 colour 0 n", n_combinations)
    elif (n_blue_per_trial[n_red]['blue'] == 1) | (n_blue_per_trial[n_red]['red'] == 1):
        n_combinations = (n_trials * (n_trials + 1)) / 2
    #         print("n combi 1 colour 1 n", n_combinations)
    else:
        n_combinations = ((n_trials - 1) * n_trials) / 2
    #         print("n combi cols >1n", n_combinations)

    n_trials_blue = n_blue_per_trial[n_red]['blue']

    p_red_trial = n_red / n_cards

    #     print("N blue trials", n_trials_blue)
    #     print("p red", p_red_trial)
    #     print("p red given trials", p_red_trial ** (n_trials - n_trials_blue))
    #     print("p blue given trials", p_blue_trial ** n_trials_blue)
    return (p_blue_trial ** n_trials_blue) * (p_red_trial ** (n_trials - n_trials_blue)) * n_combinations


def test_lie_thresholds(thresholds, trials, n_cards, gt_lies):
    precision = []
    recall = []
    accuracy = []

    for th in thresholds:
        lie_detect = []
        for i, t in enumerate(trials):
            print("TRIAL", i)
            p = check_lie_prob(n_cards, t['n_red'], t['outcome'])
            print("event prob", p)
            if p < th:
                lie_detect.append(1)
            else:
                lie_detect.append(0)
        precision.append(precision_score(gt_lies, lie_detect))
        recall.append(recall_score(gt_lies, lie_detect))
        accuracy.append(accuracy_score(gt_lies, lie_detect))

