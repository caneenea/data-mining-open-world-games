import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def elbow_plot(df, k_min=2, k_max=10, outpath="reports/figures/elbow.png"):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    ks = list(range(k_min, k_max + 1))
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method (Inertia vs K)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    return ks, inertias


def silhouette(df, k=3):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(X)

    return silhouette_score(X, labels)
