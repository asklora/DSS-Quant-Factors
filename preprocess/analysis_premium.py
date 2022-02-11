import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from global_vars import *
from general.sql_process import read_table, read_query

def avg_premium_pillar_(group='USD', weeks_to_expire=4):
    ''' calculate (absolute) average factor premiums for each pillar

    Returns
    -------
    DataFrame:  Index = factors;
                Columns = average periods
                Value = average of all factors over that periods
    '''

    query = f"SELECT p.*, f.pillar FROM {factor_premium_table} p " \
            f"INNER JOIN {formula_factors_table_prod} f ON p.field=f.name " \
            f"WHERE \"group\"='{group}' AND weeks_to_expire={weeks_to_expire}"
    df = read_query(query, db_url_read)

    avg_premium = {}
    for pillar, g in df.groupby('pillar'):
        avg_premium[pillar] = []
        for i in [1, 3, 5, 10]:
            start_date = (dt.datetime.today() - relativedelta(years=i)).date()
            g_period = g.loc[g["trading_day"] >= start_date]
            g_period_avg = g_period.groupby(["field"])["value"].mean()
            g_period_avg = g_period_avg.sort_values(ascending=False)
            g_period_avg.name = str(i)
            avg_premium[pillar].append(g_period_avg)

            g_period["value"] = g_period["value"].abs()
            g_period_abs_avg = g_period.groupby(["field"])["value"].mean()
            g_period_abs_avg = g_period_abs_avg.sort_values(ascending=False)
            g_period_abs_avg.name = f'{i}_abs'
            avg_premium[pillar].append(g_period_abs_avg)

        avg_premium[pillar] = pd.concat(avg_premium[pillar], axis=1)
    return avg_premium

def best_premium_pillar_(group='USD', weeks_to_expire=4, average_days=7):
    ''' calculate actual best factor premiums for each pillar

    Returns
    -------
    DataFrame:  Index = factors;
                Columns = average periods
                Value = average of all factors over that periods
    '''

    query = f"SELECT p.*, f.pillar FROM {factor_premium_table} p " \
            f"INNER JOIN {formula_factors_table_prod} f ON p.field=f.name " \
            f"WHERE \"group\"='{group}' AND weeks_to_expire={weeks_to_expire} AND average_days={average_days}"
    df = read_query(query, db_url_read)

    avg_premium = {}
    for pillar, g in df.groupby('pillar'):
        avg_premium[pillar] = []
        for i in [5]:
            start_date = (dt.datetime.today() - relativedelta(years=i)).date()
            g_period = g.loc[g["trading_day"] >= start_date]

            def period_part(g, th=1/3):
                conditions = [
                    (g["value"]>g["value"].quantile(th)),
                    (g["value"]<g["value"].quantile(1 - th)),
                ]
                choices = [2, 0]
                g["select"] = np.select(conditions, choices, default=1)
                g_ret = g.groupby(["select"])['value'].mean()
                g_select = g.set_index('field')['select']
                g_select = g_select.append(g_ret)
                return g_select
            g_period_ret = g_period.groupby("trading_day").apply(period_part)
            sns.heatmap(g_period_ret.iloc[:,:-3])
            plt.title(pillar)
            plt.show()
            plt.close()
            avg_premium[pillar].append(g_period_ret)
        avg_premium[pillar] = pd.concat(avg_premium[pillar], axis=1)
    print(df_period_avg)

class find_reverse:
    ''' Compare predict next period +/- using
        1. lasso
        2. logistic lasso: convert training predictions to 0(-)/1(+)
        3. moving average: if average of past n period > 0 -> +
    '''

    def __init__(self, group='USD', weeks_to_expire=4, average_days=7):
        ''' find better way to reverse premium than past 10-year average '''

        query = f"SELECT p.*, f.pillar FROM {factor_premium_table} p " \
                f"INNER JOIN {formula_factors_table_prod} f ON p.field=f.name " \
                f"WHERE \"group\"='{group}' AND weeks_to_expire={weeks_to_expire} AND average_days={average_days}"
        df = read_query(query, db_url_read).sort_values(by=['field', 'trading_day'])
        print(df.describe())

        steps = 10
        scores = {}
        methods = [self.__lasso_pred]     # self.__log_lasso_pred, self.__ma_pred
        for func in methods:
            scores[func.__name__] = {}

        for n in range(50, 160, steps):
            for func in methods:
                for n_test in range(12, 13, 12):
                    for alpha in range(3, 9, 1):
                        print(f'---> {n}')
                        samples = find_reverse.__split(df, n_x=n, n_test=n_test)
                        # scores[func.__name__][(n, n_test)] = func(samples)
                        scores[func.__name__][(n, alpha)] = func(samples, alpha=np.power(0.1, alpha))

        accuracy = []
        for k, v in scores.items():
            scores[k] = pd.DataFrame(v).transpose()
            accuracy.append(scores[k]['avg_diff'].copy())
            scores[k].reset_index()

        accuracy = pd.concat(accuracy, axis=1).unstack()
        print(accuracy)

    @staticmethod
    def __split(df, n_x, n_test):
        ''' split train / test sets:
            train = all samples of (n_x * periods -> next y)
            test = moving windows of the most recent n_test period
        '''
        date_list = sorted(list(df['trading_day'].unique()))
        for i in range(1, n_x+1):
            df[f'x_{i}'] = df.groupby(['field'])['value'].shift(i)

        samples = {}
        for i in range(1, n_test+1):
            test_period = date_list[-i]
            train = df.loc[df['trading_day']<test_period].dropna(how='any')
            test = df.loc[df['trading_day']==test_period].dropna(how='any')
            print(f'   train (n={len(train)}); test (n={len(test)})')
            if len(test) > 0:
                train_X, train_y = train.filter(regex='^x_'), train['value'].values
                test_X, test_y = test.filter(regex='^x_'), test['value'].values
                samples[test_period] = (train_X, train_y, test_X, test_y)

        return samples

    def __log_lasso_pred(self, samples):
        ''' reverse if Lasso Logistic Regression Predict Y to be positive '''

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        score = {}
        for test_period, (train_X, train_y, test_X, test_y) in samples.items():

            # convert continuous y -> 0(if < 0) / 1(if > 0)
            train_y = np.where(train_y > 0, 1, 0)
            test_y = np.where(test_y > 0, 1, 0)

            # train / predict
            log = LogisticRegression(penalty='l1', solver='liblinear')
            log.fit(train_X, train_y)
            pred = log.predict(test_X)

            # evaluate
            score[test_period] = {}
            for m in [accuracy_score, precision_score, recall_score]:
                score[test_period][m.__name__] = m(pred, test_y)
            score[test_period]['actual+%'] = (test_y==1).sum() / len(test_y)
            score[test_period]['pred+%'] = (pred==1).sum() / len(test_y)
        score_df = pd.DataFrame(score).transpose()
        return score_df.mean()

    def __lasso_pred(self, samples, alpha=0.0001):
        ''' reverse if Lasso Logistic Regression Predict Y to be positive '''

        from sklearn.linear_model import Lasso
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        score = {}
        for test_period, (train_X, train_y, test_X, test_y) in samples.items():

            # train / predict
            clf = Lasso(alpha=alpha)
            clf.fit(train_X, train_y)
            pred = clf.predict(test_X)

            score[test_period] = {"avg_diff": np.where(pred>0, test_y, -test_y).mean() - test_y.mean()}

            # convert continuous y -> 0(if < 0) / 1(if > 0)
            test_y = np.where(test_y > 0, 1, 0)
            pred = np.where(pred > 0, 1, 0)

            # evaluate
            for m in [accuracy_score, precision_score, recall_score]:
                score[test_period][m.__name__] = m(pred, test_y)
            score[test_period]['actual+%'] = (test_y==1).sum() / len(test_y)
            score[test_period]['pred+%'] = (pred==1).sum() / len(test_y)
        score_df = pd.DataFrame(score).transpose()
        return score_df.mean()

    def __ma_pred(self, samples):
        ''' reverse if Lasso Logistic Regression Predict Y to be positive '''

        from sklearn.metrics import accuracy_score, precision_score, recall_score

        score = {}
        for test_period, (train_X, train_y, test_X, test_y) in samples.items():

            # convert continuous y -> 0(if < 0) / 1(if > 0)
            test_y = np.where(test_y > 0, 1, 0)
            pred = np.where(test_X.mean(1) > 0, 1, 0)

            # evaluate
            score[test_period] = {}
            for m in [accuracy_score, precision_score, recall_score]:
                score[test_period][m.__name__] = m(pred, test_y)
            score[test_period]['actual+%'] = (test_y==1).sum() / len(test_y)
            score[test_period]['pred+%'] = (pred==1).sum() / len(test_y)
        score_df = pd.DataFrame(score).transpose()
        return score_df.mean()

if __name__ == "__main__":
    # avg_premium_pillar_()
    # best_premium_pillar_()
    find_reverse(weeks_to_expire=26, average_days=7)
