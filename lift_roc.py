# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

dat = pd.read_csv(r'D:\programGao\homework_SouthGermanCredit\dat0713.csv')
print(dat.head())
train_x, test_x, train_y, test_y = model_selection.train_test_split(dat.iloc[:, 0:-1], dat.iloc[:, -1], test_size=0.3,
                                                                    random_state=123)
# randomforest
rf0 = RandomForestClassifier(n_estimators=500,
                             criterion='gini',
                             max_depth=4,
                             min_samples_split=5,
                             min_samples_leaf=5,
                             min_weight_fraction_leaf=0.1)
rf0.fit(train_x, train_y)
# predict
rf_pred = rf0.predict(test_x)
rf_proba = rf0.predict_proba(test_x)


def lift_curve_(true_y, prob_y, group_n, if_plt):
    result = pd.DataFrame({'target': true_y, 'proba': prob_y})
    proba = result.proba.copy()
    for i in range(group_n):
        p1 = np.percentile(result.proba, i * (100 / group_n))
        p2 = np.percentile(result.proba, (i + 1) * (100 / group_n))
        proba[(result.proba >= p1) & (result.proba <= p2)] = (i + 1)
    result['grade'] = proba
    bad = result.groupby(by=['grade']).target.sum()
    tot = result.groupby(by=['grade']).grade.count()
    df_agg = pd.concat([bad, tot], axis=1)
    df_agg.columns = ['bad', 'tot']
    df_agg = df_agg.sort_index(ascending=False).reset_index()
    df_agg['bad_rate'] = df_agg['bad']/df_agg['tot']
    df_agg['bad_r'] = df_agg['bad']/sum(df_agg['bad'])
    df_agg['tot_r'] = df_agg['tot']/sum(df_agg['tot'])
    bad_list = []
    tot_list = []
    a = 0
    b = 0
    for i in range(df_agg.shape[0]):
        if i == 0:
            a = df_agg['bad_r'].to_list()[i]
            b = df_agg['tot_r'].to_list()[i]
            bad_list.append(a)
            tot_list.append(b)
        else:
            a += df_agg['bad_r'].to_list()[i]
            b += df_agg['tot_r'].to_list()[i]
            bad_list.append(a)
            tot_list.append(b)
    df_agg['cum_bad'] = bad_list
    df_agg['cum_tot'] = tot_list
    df_agg['lift'] = df_agg['cum_bad']/df_agg['cum_tot']
    if if_plt is True:
        plt.figure(figsize=(4, 3))
        plt.plot(np.arange(0, 1.0, 1/len(df_agg['lift'])), df_agg['lift'].to_list())
        plt.title('Lift Curve')
        plt.grid(True)
        plt.show()

    return df_agg


if __name__ == '__main__':
    lift_curve_(test_y, rf_proba[:, 1], 20, if_plt=True)
