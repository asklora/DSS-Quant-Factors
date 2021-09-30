import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def count_plot():
    for i in [91, 30, 7]:
        # csv_name = f'new2_stepwise_{i}.csv'
        # csv_name = f'hierarchy_average_vol_{i}_cophenetic.csv'
        csv_name = f'hierarchy_cluster_{i}_cophenetic.csv'
        df = pd.read_csv(csv_name)

        df['factors'] = df['factors'].str.split(', ')
        df['n_factors'] = df['factors'].str.len()
        df['dummy'] = True

        for name, g in df.groupby(['dummy']):
            f = [e for x in g['factors'].values for e in x]
            x = dict(Counter(f))
            x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

            plt.barh(y=list(x.keys()), width=list(x.values()))
            plt.title(csv_name)
            plt.tight_layout()
            plt.show()
            print(df)

if __name__=="__main__":
    count_plot()