import matplotlib.pyplot as plt
import global_vars
import pandas as pd
from collections import Counter


def results_analysis():
    df = pd.read_csv('stepwise_cluster_FCM_new_30.csv').sort_values(by=['xie_beni_index'])

    x = df.iloc[:50, -1].str.split(', ', expand=True).values[:, 3:].flatten()
    print(Counter(x))
    exit(1)

    plt.plot(df['xie_beni_index'].values[:150])
    plt.show()
    print(df)

if __name__ == "__main__":
    results_analysis()

