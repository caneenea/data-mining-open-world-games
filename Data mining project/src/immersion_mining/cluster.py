import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def fit_kmeans(df: pd.DataFrame, k: int = 3, seed: int = 42):
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = model.fit_predict(X)

    return model, scaler, labels


def cluster_summary(df: pd.DataFrame, labels):
    out = df.copy()
    out["cluster"] = labels
    return out.groupby("cluster").mean(numeric_only=True).round(2)
