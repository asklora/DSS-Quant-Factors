

def test_cluster(method='dbscan'):

    from sklearn.cluster import KMeans, DBSCAN, OPTICS
    from sklearn.decomposition import PCA
    from sklearn import metrics
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn import decomposition

    # We have to fill all the nans with 1(for multiples) since Kmeans can't work with the data that has nans.
    X = factors.copy(1)[col_list].transpose().values
    X = np.nan_to_num(X, -99.9)
    X = StandardScaler().fit_transform(X)

    if method == 'kmean':
        cluster_no = 5
        kmeans = KMeans(cluster_no).fit(X)
        y = kmeans.predict(X)
    elif method == 'optics':
        opt = OPTICS(min_samples=1).fit(X)
        y = opt.labels_
    elif method == 'dbscan':
        db = DBSCAN(min_samples=3, eps=0.9).fit(X)
        y = db.labels_
        print(y)

    # use PCA and plot
    ppca = PCA(n_components=2).fit(X)
    X_pca = ppca.transform(X)
    plt.figure(figsize=(6, 6), dpi=120, constrained_layout=True)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=50, cmap='viridis')
    for i in range(len(col_list)):
        plt.annotate(col_list[i], (X_pca[i, 0], X_pca[i, 1]), fontsize=4)
    plt.savefig(f'eda/cluster_{method}.png')

    # calculate matrices for clustering
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]

    # save excel for pillar difference
    dic = formula[['pillar','name']].set_index(['name'])['pillar'].to_dict()
    cluster_df = pd.DataFrame({'name':col_list, 'cluster':y})
    cluster_df['pillar'] = cluster_df['name'].map(dic)
    cluster_df.sort_values(by=['pillar']).to_csv(f'eda/cluster_{method}.csv', index=False)

    # calculate matrics
    m = {}
    for i in clustering_metrics:
        pillar_code = cluster_df['pillar'].map({'momentum':0, 'quality':1, 'value':2}).to_list()
        m[i.__name__] = i(cluster_df['cluster'].to_list(), pillar_code)
    print(m)

if __name__ == "__main__":
    test_cluster()