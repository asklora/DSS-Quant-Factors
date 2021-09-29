import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def count_plot():
    df = pd.read_csv('new_stepwise_30.csv')
    df['factors'] = df['factors'].str.split(', ')
    df['n_factors'] = df['factors'].str.len()

    f = [e for x in df['factors'].values for e in x]
    x = dict(Counter(f))
    x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

    plt.barh(y=list(x.keys()), width=list(x.values()))
    plt.tight_layout()
    plt.show()
    print(df)



if __name__=="__main__":
    count_plot()