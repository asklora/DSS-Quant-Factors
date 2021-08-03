# Index using SPX/HSI/CSI300 returns
# stock_return_col = ['stock_return_r1_0', 'stock_return_r6_2', 'stock_return_r12_7']
# major_index = ['period_end','.SPX','.HSI','.CSI300']    # try include major market index first
# index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]
# index_ret = index_ret.set_index(['period_end', 'ticker']).unstack()
# index_ret.columns = [f'{x[1]}_{x[0][13:]}' for x in index_ret.columns.to_list()]
# index_ret = index_ret.reset_index()
# index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])

def download_org_ratios(use_biweekly_stock, stock_last_week_avg, method='mean', change=True):
    ''' download the aggregated value of all original ratios by each group '''

    db_table = global_vals.processed_group_ratio_table
    if stock_last_week_avg:
        db_table += '_weekavg'
    elif use_biweekly_stock:
        db_table += '_biweekly'

    with global_vals.engine_ali.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {db_table} WHERE method = '{method}'", conn)
    global_vals.engine_ali.dispose()
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y-%m-%d')
    field_col = df.columns.to_list()[2:-1]

    if change:  # calculate the change of original ratio from T-1 -> T0
        df[field_col] = df[field_col]/df.sort_values(['period_end']).groupby(['group'])[field_col].shift(1)-1
        df[field_col] = df[field_col].apply(trim_outlier)

    df.columns = df.columns.to_list()[:2] + ['org_'+x for x in field_col] + [df.columns.to_list()[-1]]

    return df.iloc[:,:-1]

# 3. (Removed) Add original ratios variables
# org_df = download_org_ratios(use_biweekly_stock, stock_last_week_avg)
# non_factor_inputs = org_df.merge(non_factor_inputs, on=['period_end'], how='outer')

def fft_combine():
    df,_,_ = combine_data(use_biweekly_stock=True, stock_last_week_avg=False)  # combine all data
    y = df.loc[(df['group']=='101010'), ['period_end', 'book_to_price']]
    y = y['book_to_price'].values
    yf = fft(y)
    from scipy.signal import blackman
    import matplotlib.pyplot as plt

    N = len(y)
    T = 1/3

    w = blackman(N)
    ywf = fft(y * w)
    # ryf = rfft(y)

    xf = fftfreq(N, T)

    plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
    plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
    plt.legend(['FFT', 'FFT w. window'])
    plt.grid()
    plt.show()
    exit(0)
