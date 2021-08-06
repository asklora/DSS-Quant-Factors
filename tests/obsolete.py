# Index using SPX/HSI/CSI300 returns
# stock_return_col = ['stock_return_r1_0', 'stock_return_r6_2', 'stock_return_r12_7']
# major_index = ['period_end','.SPX','.HSI','.CSI300']    # try include major market index first
# index_ret = index_ret.loc[index_ret['ticker'].isin(major_index)]
# index_ret = index_ret.set_index(['period_end', 'ticker']).unstack()
# index_ret.columns = [f'{x[1]}_{x[0][13:]}' for x in index_ret.columns.to_list()]
# index_ret = index_ret.reset_index()
# index_ret['period_end'] = pd.to_datetime(index_ret['period_end'])





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
