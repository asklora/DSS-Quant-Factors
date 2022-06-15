import numpy as np
import pandas as pd
import datetime as dt
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
import math

from global_vars import *
from general.sql.sql_process import read_table, read_query

logger = logger(__name__, LOGGER_LEVEL)


def download_clean_macros():
    ''' download macros data from DB and preprocess: convert some to yoy format '''

    logger.info(f'Download macro data from {macro_data_table}')

    # combine macros & vix data
    query = f"SELECT * FROM {macro_data_table} "
    # query += f"WHERE field in (SELECT our_name FROM {ingestion_name_macro_table} WHERE is_active)"
    macros = read_query(query)

    vix = read_table(vix_data_table, db_url_read)
    vix = vix.rename(columns={"vix_id": "field", "vix_value": "value"})
    macros = macros.append(vix)

    fred = read_table(fred_data_table)
    fred['field'] = "fred_data"
    macros = macros.append(fred)

    macros = macros.pivot(index=["trading_day"], columns=["field"], values="value").reset_index()
    macros['trading_day'] = pd.to_datetime(macros['trading_day'], format='%Y-%m-%d')
    macros = macros.set_index("trading_day")
    macros = macros.resample('D').ffill()
    macros = macros.reset_index()

    yoy_col = macros.select_dtypes('float').columns[macros.select_dtypes('float').mean(axis=0) > 80]  # convert YoY
    num_col = macros.select_dtypes('float').columns.to_list()  # all numeric columns

    # update yoy ratios
    macros_yoy = macros[["trading_day"] + list(yoy_col)]
    macros_yoy["trading_day"] = macros_yoy["trading_day"].apply(lambda x: x + relativedelta(years=1))
    macros = macros.merge(macros_yoy, on=["trading_day"], how="left", suffixes=("", "_1yb"))
    macros = macros.sort_values(by="trading_day").fillna(method='ffill')
    for i in yoy_col:
        macros[i] = macros[i] / macros[i + "_1yb"] - 1
    macros = macros.dropna(subset=num_col, how="all")

    return macros[["trading_day"] + list(num_col)].drop_duplicates("trading_day")


def download_index_return():
    ''' download index return data from DB and preprocess: convert to YoY and pivot table '''

    logger.info(f'Download index return data from [{processed_ratio_table}]')

    # read stock return from ratio calculation table
    index_col = ['stock_return_r12_7', 'stock_return_r1_0', 'stock_return_r6_2']
    index_query = f"SELECT * FROM {processed_ratio_table} WHERE ticker like '.%%' AND field in {tuple(index_col)}"
    index_ret = read_query(index_query, db_url_read)

    index_ret = index_ret.pivot(index=["ticker", "trading_day"], columns=["field"], values="value").reset_index()

    # Index using all index return12_7, return6_2 & vol_30_90 for 6 market based on num of ticker
    major_index = ['trading_day', '.SPX', '.CSI300', '.SXXGR', '.HSI']  # try include 3 major market index first
    index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]

    index_ret = index_ret.set_index(['trading_day', 'ticker'])[index_col].unstack()
    index_ret.columns = [f'{x[1]}_{x[0]}' for x in index_ret.columns.to_list()]
    index_ret = index_ret.reset_index()
    index_ret['trading_day'] = pd.to_datetime(index_ret['trading_day'])

    return index_ret


def combine_data(weeks_to_expire, update_since=None, trim=False):
    """ combine factor premiums with ratios """

    # Read premium sql from different tables
    factor_table_name = factor_premium_table

    logger.info(f'Use [{weeks_to_expire}] week premium')
    conditions = ['"group" IS NOT NULL',
                  f"weeks_to_expire={weeks_to_expire}"]

    if isinstance(update_since, datetime):
        update_since_str = update_since.strftime(r'%Y-%m-%d %H:%M:%S')
        conditions.append(f"trading_day >= TO_TIMESTAMP('{update_since_str}', 'YYYY-MM-DD HH:MI:SS')")

    prem_query = f"SELECT * FROM {factor_table_name} WHERE {' AND '.join(conditions)};"
    df = read_query(prem_query, db_url_read)
    df = df.pivot(index=['trading_day', 'group', 'average_days'], columns=['field'], values="value")

    trim_cols = df.filter(regex='^trim_').columns.to_list()
    if trim:
        df = df[trim_cols]
    else:
        df = df.drop(trim_cols)

    # remove null > 50% sample
    df = df.loc[df.isnull().sum(axis=1) < df.shape[1]/2]
    df = df.T.fillna(df.T.mean()).T
    df.columns.name = None
    df = df.reset_index()
    df['trading_day'] = pd.to_datetime(df['trading_day'], format='%Y-%m-%d')  # convert to datetime

    # read formula table
    formula = read_table(factors_formula_table, db_url_read)
    formula = formula.loc[formula['name'].isin(df.columns.to_list())]  # filter existing columns from factors

    # Research stage using 10 selected factor only
    x_col = {'factor': formula['name'].to_list()}

    for p in formula['pillar'].unique():
        x_col[p] = formula.loc[formula['pillar'] == p, 'name'].to_list()  # factor for each pillar

    # df = df.loc[df['trading_day'] < dt.datetime.today() + MonthEnd(-2)]  # remove records within 2 month prior to today

    # 1. Add Macroeconomic variables - from Datastream
    macros = download_clean_macros()
    x_col['macro'] = macros.columns.to_list()[1:]  # add macros variables name to x_col

    # 2. Add index return variables
    index_ret = download_index_return()
    x_col['index'] = index_ret.columns.to_list()[1:]  # add index variables name to x_col

    # Combine non_factor_inputs and move it 1-month later -> factor premium T0 assumes we knows price as at T1
    # Therefore, we should also know other data (macro/index/group fundamental) as at T1
    index_ret["trading_day"] = pd.to_datetime(index_ret["trading_day"])
    macros["trading_day"] = pd.to_datetime(macros["trading_day"])
    non_factor_inputs = macros.merge(index_ret, on=['trading_day'], how='outer')
    non_factor_inputs['trading_day'] = non_factor_inputs['trading_day'].apply(
        lambda x: x - relativedelta(weeks=weeks_to_expire))
    df = df.merge(non_factor_inputs, on=['trading_day'], how='outer').sort_values(['group', 'trading_day'])
    df[x_col['macro'] + x_col['index']] = df.sort_values(["trading_day", "group"]).groupby(["group"])[
        x_col['macro'] + x_col['index']].ffill()
    df = df.dropna(subset=["group"])

    # use only period_end date
    indexes = pd.MultiIndex.from_product([df['group'].unique(), df['trading_day'].unique()],
                                         names=['group', 'trading_day']).to_frame().reset_index(drop=True)
    df = pd.merge(df, indexes, on=['group', 'trading_day'], how='right')
    logger.info(f"Factors: {x_col['factor']}")

    return df.sort_values(by=['group', 'trading_day']), x_col['factor'], x_col


class load_data:
    """ main function:
        1. split train + valid + tests -> sample set
        2. convert x with standardization, y with qcut """

    def __init__(self, weeks_to_expire, update_since=None, trim=False):
        """ combine all possible data to be used

        Parameters
        ----------
        weeks_to_expire : text
        update_since : bool, optional

        """

        # define self objects
        self.pred_currency = None
        self.train_currency = None
        self.sample_set = {}
        self.group = pd.DataFrame()
        self.weeks_to_expire = weeks_to_expire
        self.main, self.factor_list, self.x_col_dict = combine_data(weeks_to_expire,
                                                                    update_since=update_since,
                                                                    trim=trim)  # combine all data
        # print(self.main)

    def split_train_currency(self, train_currency, pred_currency, average_days, **kwargs):
        """ split main sample sets in to industry_parition or country_partition """

        if train_currency == 'currency':
            self.train_currency = ['CNY', 'HKD', 'EUR', 'USD']
        else:
            self.train_currency = train_currency.split(',')
        self.pred_currency = pred_currency.split(',')

        all_currency = list(set(self.pred_currency + self.train_currency))

        # train on currency partition factors
        self.group = self.main.loc[(self.main['group'].isin(all_currency)) &
                                   (self.main['average_days'] == average_days)]

        # calculate y for all factors
        df_y = self.group[['group', 'trading_day'] + self.x_col_dict['factor']].rename(
            columns={x: 'y_' + x for x in self.x_col_dict['factor']})
        df_y["trading_day"] = df_y['trading_day'].apply(lambda x: x - relativedelta(weeks=self.weeks_to_expire))
        self.group = self.group.merge(df_y, on=["group", "trading_day"], how="outer")

    @staticmethod
    def y_replace_median(_y_qcut, arr, arr_cut, arr_test, arr_test_cut):
        ''' convert qcut results (e.g. 012) to the median of each group for regression '''

        df = pd.DataFrame(np.vstack((arr, arr_cut))).T  # concat original & qcut
        median = df.groupby([1]).median().sort_index()[0].to_list()  # find median of each group
        arr_cut_median = pd.DataFrame(arr_cut).replace(range(_y_qcut), median)[0].values
        arr_test_cut_median = pd.DataFrame(arr_test_cut).replace(range(_y_qcut), median)[0].values
        return arr_cut_median, arr_test_cut_median

    # def neg_factor_best_period(self, df, x_col):
    #
    #     best_best = {}
    #     for name in x_col:
    #         best = {}
    #         g = df[name]
    #         for i in np.arange(12, 120, 12):
    #             g['ma'] = g.rolling(i, min_periods=1, closed='left')['premium'].mean()
    #             g['new_premium'] = np.where(g['ma'] >= 0, g['premium'], -g['premium'])
    #             best[i] = g['new_premium'].mean()
    #
    #         best_best[name] = [k for k, v in best.items() if v == np.max(list(best.values()))][0]
    #
    #     return best_best

    def y_qcut_all(self, _y_qcut, _factor_reverse, factor_list, defined_cut_bins=[], use_median=True, **kwargs):
        """ convert continuous Y to discrete (0, 1, 2) for all factors during the training / testing period """

        null_col = self.train.isnull().sum()
        null_col = list(null_col.loc[(null_col == len(self.train))].index)  # remove null col from y col
        y_col = ['y_' + x for x in factor_list if x not in null_col]
        cut_col = [x + "_cut" for x in y_col]

        # convert consistently negative premium factor to positive
        self.__y_convert_neg_factors(_factor_reverse=_factor_reverse)

        # use n-split qcut median as Y
        if _y_qcut > 0:
            arr = self.train[y_col].values.flatten()  # Flatten all training factors to qcut all together
            # arr[(arr>np.quantile(np.nan_to_num(arr), 0.99))|(arr<np.quantile(np.nan_to_num(arr), 0.01))] = np.nan

            if defined_cut_bins == []:
                # cut original series into bins
                arr_cut, cut_bins = pd.qcut(arr, q=_y_qcut, retbins=True, labels=False)
                cut_bins[0], cut_bins[-1] = [-np.inf, np.inf]
            else:
                # use pre-defined cut_bins for cut (since all factor should use same cut_bins)
                cut_bins = defined_cut_bins
                arr_cut = pd.cut(arr, bins=cut_bins, labels=False)

            arr_test = self.test[y_col].values.flatten()  # Flatten all testing factors to qcut all together
            arr_test_cut = pd.cut(arr_test, bins=cut_bins, labels=False)

            if use_median:  # for regression -> remove noise by regression on median of each bins
                arr_cut, arr_test_cut = load_data.y_replace_median(_y_qcut, arr, arr_cut, arr_test, arr_test_cut)

            self.train[cut_col] = np.reshape(arr_cut, (len(self.train), len(y_col)), order='C')
            self.test[cut_col] = np.reshape(arr_test_cut, (len(self.test), len(y_col)), order='C')

            # write cut_bins to DB
            self.cut_bins_df = self.train[cut_col].apply(pd.value_counts).transpose()
            self.cut_bins_df = self.cut_bins_df.divide(self.cut_bins_df.sum(axis=1).values, axis=0).reset_index()
            self.cut_bins_df['negative'] = False
            # self.cut_bins_df.loc[self.cut_bins_df['index'].isin([x+'_cut' for x in neg_factor]), 'negative'] = True
            self.cut_bins_df['pillar'] = [x[2:-4] for x in self.cut_bins_df['index']]
            self.cut_bins_df['cut_bins_low'] = cut_bins[1]
            self.cut_bins_df['cut_bins_high'] = cut_bins[-2]
        else:
            self.train[cut_col] = self.train[y_col]
            self.test[cut_col] = self.test[y_col]

        return y_col

    def __y_convert_neg_factors(self, _factor_reverse=False, n_years=7, lasso_alpha=1e-5):
        '''  convert consistently negative premium factor to positive
        refer to: https://loratechai.atlassian.net/wiki/spaces/SEAR/pages/994803719/AI+Score+Brainstorms+2022-01-28

        Parameters
        ----------
        _factor_reverse (Bool, Optional):
            if True, use training period average; else (default) use lasso
        n_years (Int, Optional):
            lasso training period length (default = 7) years
        lasso_alpha (Float, Optional):
            lasso training alpha lasso_alpha (default = 1e-5)
        '''

        y_col = read_query(f'SELECT name FROM {factors_formula_table} WHERE is_active')['name'].to_list()

        if type(_factor_reverse) == type(None):
            self.neg_factor = []
        elif _factor_reverse:  # using average of training period -> next period +/-
            m = self.train.filter(y_col).mean(axis=0)
            self.neg_factor = list(m[m < 0].index)
        else:
            start_date = dt.datetime.today() - relativedelta(years=n_years)
            n_x = len(self.train.loc[self.train['trading_day'] >= start_date, 'trading_day'].unique())

            train_X = self.train.set_index('trading_day').filter(y_col).stack().reset_index()
            train_X.columns = ['trading_day', 'field', 'y']

            for i in range(1, n_x + 1):
                train_X[f'x_{i}'] = train_X.groupby(['field'])['y'].shift(i)
            train_X = train_X.dropna(how='any')
            clf = Lasso(alpha=lasso_alpha)
            clf.fit(train_X.filter(regex='^x_').values, train_X['y'].values)

            test_X = train_X.groupby('field').last().reset_index()
            test_X['pred'] = clf.predict(test_X.filter(regex='^x_|^y$').values[:, :-1])
            self.neg_factor = test_X.loc[test_X['pred'] < 0, 'field'].to_list()

        # self.neg_factor = []
        self.train[self.neg_factor + ['y_' + x for x in self.neg_factor]] *= -1
        self.test[self.neg_factor + ['y_' + x for x in self.neg_factor]] *= -1

    def split_train_test(self, testing_period,
                         _factor_pca=None, mi_pca=0.6, train_lookback_year=20,
                         ar_period=[], ma3_period=[], ma12_period=[],
                         **kwargs):
        """ split training / testing set based on testing period """

        current_train_currency = self.group.copy(1)
        start = testing_period - relativedelta(years=train_lookback_year)  # train df = 20*12 months

        # factor with ARMA history as X (if using pca, all history first)
        arma_col = self.x_col_dict['factor'] + self.x_col_dict['index'] + self.x_col_dict['macro']

        # 1. [Prep X] Calculate the time_series history for predicted Y (use 1/2/12 based on ARIMA results)
        self.x_col_dict['ar'] = []
        for i in ar_period:
            ar_col = [f"ar_{x}_{i}m" for x in arma_col]
            current_train_currency[ar_col] = current_train_currency.groupby(['group'])[arma_col].shift(i)
            self.x_col_dict['ar'].extend(ar_col)  # add AR variables name to x_col

        # 2. [Prep X] Calculate the moving average for predicted Y
        self.x_col_dict['ma'] = []
        for i in ma3_period:  # include moving average of 3-5, 6-8, 9-11
            ma_q = current_train_currency.groupby(['group'])[arma_col].rolling(3, min_periods=1).mean().reset_index(level=0,
                                                                                                           drop=True)
            ma_q_col = ma_q.columns = [f"ma_{x}_q" for x in arma_col]
            current_train_currency = pd.concat([current_train_currency, ma_q], axis=1)
            current_train_currency[[f'{x}{i}' for x in ma_q_col]] = current_train_currency.groupby(['group'])[ma_q_col].shift(i)
            self.x_col_dict['ma'].extend([f'{x}{i}' for x in ma_q_col])  # add MA variables name to x_col
        for i in ma12_period:  # include moving average of 12 - 23
            ma_y = current_train_currency.groupby(['group'])[arma_col].rolling(12, min_periods=1).mean().reset_index(level=0,
                                                                                                            drop=True)
            ma_y_col = ma_y.columns = [f"ma_{x}_y" for x in arma_col]
            current_train_currency = pd.concat([current_train_currency, ma_y], axis=1)
            current_train_currency[[f'{x}{i}' for x in ma_y_col]] = current_train_currency.groupby(['group'])[ma_y_col].shift(i)
            self.x_col_dict['ma'].extend([f'{x}{i}' for x in ma_y_col])  # add MA variables name to x_col

        # 3. [Split training/testing] sets based on testing_period
        train_end_date = testing_period - relativedelta(weeks=self.weeks_to_expire)     # avoid data snooping
        self.train = current_train_currency.loc[(start <= current_train_currency['trading_day']) &
                                       (current_train_currency['trading_day'] <= train_end_date) &
                                       (current_train_currency['group'].isin(self.train_currency))].reset_index(drop=True).copy()
        self.test = current_train_currency.loc[(current_train_currency['trading_day'] == testing_period) &
                                      (current_train_currency['group'].isin(self.pred_currency))].reset_index(drop=True).copy()

        # 4. [Prep Y]: qcut/cut for all factors to be predicted (according to factor_formula table in DB) at the same time
        self.y_col = self.y_qcut_all(**kwargs)
        self.train = self.train.dropna(subset=self.y_col, how='any').reset_index(drop=True)  # remove training sample with NaN Y

        # if using feature selection with PCA
        arma_factor = [x for x in self.x_col_dict['ar'] + self.x_col_dict['ma'] for f in self.x_col_dict['factor'] if f in x]

        # 5. [Prep X] use PCA on all Factor + ARMA inputs
        factor_pca_col = self.x_col_dict['factor'] + arma_factor
        factor_pca_train, factor_pca_test, factor_feature_name = load_data.standardize_pca_x(self.train[factor_pca_col], self.test[factor_pca_col], _factor_pca)
        self.x_col_dict['arma_pca'] = ['arma_' + str(x) for x in factor_feature_name]
        self.train[self.x_col_dict['arma_pca']] = factor_pca_train
        self.test[self.x_col_dict['arma_pca']] = factor_pca_test
        logger.info(f"After {_factor_pca} PCA [Factors]: {len(factor_feature_name)}")

        # 6. [Prep X] use PCA on all index/macro inputs
        group_index = {"USD": ".SPX", "HKD": ".HSI", "EUR": ".SXXGR", "CNY": ".CSI300"}
        mi_pca_col = []
        for train_cur in self.train_currency:
            mi_pca_col += [x for x in self.x_col_dict['index'] if re.match(f'^{group_index[train_cur]}', x)]
        mi_pca_col += self.x_col_dict['macro']
        mi_pca_train, mi_pca_test, mi_feature_name = load_data.standardize_pca_x(self.train[mi_pca_col], self.test[mi_pca_col], mi_pca)
        self.x_col_dict['mi_pca'] = ['mi_' + str(x) for x in mi_feature_name]
        self.train[self.x_col_dict['mi_pca']] = mi_pca_train
        self.test[self.x_col_dict['mi_pca']] = mi_pca_test
        logger.info(f"After {mi_pca} PCA [Macro+Index]: {len(mi_feature_name)}")

        def divide_set(df):
            """ split x, y from main """
            x_col = self.x_col_dict['arma_pca'] + self.x_col_dict['mi_pca']
            y_col_cut = [x + '_cut' for x in self.y_col]
            return df.filter(x_col).values, df[self.y_col].values, df[y_col_cut].values, \
                   df.filter(x_col).columns.to_list()

        self.sample_set['train_x'], self.sample_set['train_y'], self.sample_set['train_y_final'], _ = divide_set(self.train)
        self.sample_set['test_x'], self.sample_set['test_y'], self.sample_set['test_y_final'], self.x_col = divide_set(self.test)

    @staticmethod
    def standardize_pca_x(X_train, X_test, n_components=None):
        ''' standardize x + PCA applied to x with train_x fit '''

        org_feature = X_train.columns.to_list()
        X_train, X_test = X_train.values, X_test.values
        if (type(n_components) == type(None)) or (math.isnan(n_components)):
            scaler = StandardScaler()
        else:
            scaler = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=n_components))])
            X_train = np.nan_to_num(X_train, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)

        scaler.fit(X_train)
        x_train = scaler.transform(X_train)
        x_test = scaler.transform(X_test)
        if n_components:
            feature_name = range(1, x_train.shape[1] + 1)
        else:
            feature_name = org_feature
        return x_train, x_test, feature_name

    def split_valid(self, _valid_pct, _valid_method, **kwargs):
        """ split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set

        Parameters
        ----------
        testing_period :
            testing set starting date.
        _valid_pct (Float):
            must be < 1, use _valid_pct% of training as validation.
        _valid_method (Str / Int):
            if "cv", cross validation between groups;
            if "chron", use last 2 years / % of training as validation;
            if Int (must be year after 2008), use valid_pct% sample from year indicated by valid_method.
        """

        assert _valid_pct < 1

        # split validation set by cross-validation 5 split
        if _valid_method == "cv":
            gkf = GroupShuffleSplit(n_splits=int(round(1/_valid_pct)), random_state=666).split(
                self.sample_set['train_x'], self.sample_set['train_y_final'], groups=self.train['group'])

        # split validation set by chronological order
        elif _valid_method == "chron":
            gkf = []
            valid_len = (self.train['trading_day'].max() - self.train['trading_day'].min()) * _valid_pct
            valid_period = self.train['trading_day'].max() - valid_len
            valid_index = self.train.loc[self.train['trading_day'] >= valid_period].index.to_list()
            train_index = self.train.loc[self.train['trading_day'] < valid_period].index.to_list()
            gkf.append((train_index, valid_index))

        # valid_method can be year name (e.g. 2010) -> use _valid_pct% amount of data since 2010-01-01 as valid sets
        elif isinstance(_valid_method, int):
            valid_len = (self.train['trading_day'].max() - self.train['trading_day'].min()) * _valid_pct
            valid_start = dt.datetime(_valid_method, 1, 1, 0, 0, 0)
            valid_end = valid_start + valid_len
            valid_index = self.train.loc[(self.train['trading_day'] >= valid_start) &
                                         (self.train['trading_day'] < valid_end)].index.to_list()

            # half of valid sample have data leak from training sets
            train_index = self.train.loc[(self.train['trading_day'] < (valid_start - valid_len / 2)) |
                                         (self.train['trading_day'] >= valid_end)].index.to_list()
            gkf = [(train_index, valid_index)]
        else:
            raise ValueError(f"Invalid '_valid_method'. "
                             f"Expecting 'cv' or 'chron' or Int of year (e.g. 2010) got {_valid_method}")
        return gkf

    def split_all(self, testing_period, **kwargs):
        """ work through cleansing process """

        self.split_train_test(testing_period, **kwargs)  # split x, y for tests / train samples
        gkf = self.split_valid(**kwargs)  # split for cross validation in groups
        return self.sample_set, gkf


if __name__ == '__main__':
    pass
