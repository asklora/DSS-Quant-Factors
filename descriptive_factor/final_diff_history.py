import global_vals
import pandas as pd
import datetime as dt
from utils_des import read_item_df, feature_hierarchical_plot, selection_feature_hierarchical, \
    cluster_fcm, cluster_gaussian, cluster_hierarchical, report_to_slack, plot_scatter_2d

def test_techinical(currency='USD'):

    item_df = pd.read_csv(f'dcache_sample_91.csv')
    item_df = item_df.sort_values(by=['ticker', 'trading_day'])
    item_df_org = item_df.copy(1)

    item_df = item_df.loc[item_df['ticker'].str[0] != '.']

    if currency == 'HKD':
        item_df = item_df.loc[item_df['ticker'].str[-3:] == '.HK']
    else:
        item_df = item_df.loc[item_df['ticker'].str[-3:] != '.HK']

    miss_df = item_df.isnull().sum() / len(item_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(miss_df[miss_df > 0].sort_values(ascending=False))

    item_df.iloc[:, 2:] = __trim_outlier_std(item_df.iloc[:, 2:].fillna(0), plot=plot).values
    peroid_list = sorted(list(item_df['trading_day'].unique()))
    peroid_list.reverse()
    orginal_cols = item_df.columns.to_list()[2:]

def __trim_outlier_std(df, plot=False):
    ''' trim outlier on testing sets '''

    def trim_scaler(x):
        # s = skew(x)
        # if (s < -5) or (s > 5):
        #     x = np.log(x + 1 - np.min(x))
        x = x.values
        # m = np.median(x)
        # std = np.nanstd(x)
        # x = np.clip(x, m - 2 * std, m + 2 * std)
        return quantile_transform(np.reshape(x, (x.shape[0], 1)), output_distribution='normal', n_quantiles=1000)[:,0]

    cols = df.select_dtypes(float).columns.to_list()
    for col in cols:
        if plot:
            fig = plt.figure(figsize=(8, 4), dpi=60, constrained_layout=True)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.hist(df[col], bins=20)
        if col != 'icb_code':
            x = trim_scaler(df[col])
        else:
            x = df[col].values
        x = scale(x.T)
        # x = minmax_scale(x)
        df[col] = x
        if plot:
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.hist(df[col], bins=20)
            plt.suptitle(col)
            plt.show()
            plt.close(fig)
    return df

if __name__=="__main__":
    test_year()