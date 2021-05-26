# plotting functions for analysis and outcome reporting

import seaborn as sns
from matplotlib import ticker
import pandas as pd


def create_df_subject(trials, i):
    dict_subject = {
        "trial": range(1,41),
        "n_red": trials.n_red,
        "n_blue": 5-trials.n_red,
        "opponent_report": trials.outcome,
        "expectation_violation": trials.exp_violation,
        "random_pick_colour": [transdict[x] for x in randomPickColour[i]],
        "reported_pick_colour": reportColour[i],
        "honest_card_report": reportHonesty[i],
        "trial_result": transGameResults[i],
        "honesty_rating": honestyRatings[i],
        "RT_honesty": RThonesty[i],
        "normalized_reversed_honesty_rating": normalized_suspicion_ratings[i]
    }

    df_subject = pd.DataFrame(dict_subject)
    df_subject['surprise'] = abs(df_subject['expectation_violation'])
    df_subject['plot_partner_report'] = 1.1
    df_subject['plot_subject_report'] = -0.1
    df_subject['plot_random_pick_colour'] = -0.15
    df_subject['lied'] = df_subject['random_pick_colour'] != df_subject['reported_pick_colour']
    df_subject['partner_result'] = [1 if x == 0 else 0 if x == 1 else x for x in df_subject.trial_result.values]
    return df_subject


def create_plot_subject(df_subject, i):
    sns.set(rc={'figure.figsize':(16,7)})
    p = sns.lineplot(data=df_subject, x='trial', y='normalized_reversed_honesty_rating')
    p.set(ylim=(-0.25,1.2), xlim=(0, 41))
    p.set_ylabel('actual normalized reversed honesty ratings')
    p.set_title('Cards game progression subject '+str(i+1))

    markers2 = {"loss": "X", "win": "o", "tie": "D"}
    colors = {-1: 'red', 1: 'blue'}

    p2 = sns.lineplot(data=df_subject, x='trial', y='surprise')
    s = sns.scatterplot(data=df_subject, x='trial',y='plot_random_pick_colour', c=[colors[x] for x in df_subject.random_pick_colour])
    s1 = sns.scatterplot(data=df_subject, x='trial',y='plot_subject_report', c=[colors[x] for x in df_subject.reported_pick_colour])
    s2 = sns.scatterplot(data=df_subject, x='trial',y='plot_partner_report', c=[colors[x] for x in df_subject.opponent_report])
    s3 = sns.scatterplot(data=df_subject, x='trial',y='trial_result', markers=markers2, style=df_subject.trial_result)
    s3.xaxis.set_major_locator(ticker.MultipleLocator(1))

    s4 = sns.scatterplot(data=df_subject, x='trial', y=[-0.2 if x else None for x in df_subject.lied])
