import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pca_scatter(df: pd.DataFrame, labels, outpath: str):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(df.values)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title("Playstyle Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_cluster_profiles(summary_df: pd.DataFrame, outpath: str):
    plt.figure(figsize=(10, 4))
    summary_df.T.plot(kind="bar")
    plt.title("Cluster Feature Profiles (Mean by Cluster)")
    plt.xlabel("Feature")
    plt.ylabel("Mean (original scale)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
