import sys
import os

# Allow Python to find src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from immersion_mining.generate_data import SynthConfig, generate_synthetic_players
from immersion_mining.cluster import fit_kmeans, cluster_summary
from immersion_mining.viz import (
    ensure_dir,
    save_pca_scatter,
    save_cluster_profiles,
)
from immersion_mining.evaluate import elbow_plot, silhouette


def main():
    # Create output folders
    ensure_dir("data/raw")
    ensure_dir("reports/figures")

    # -------------------------
    # 1. Generate synthetic data
    # -------------------------
    df = generate_synthetic_players(
        SynthConfig(n_players=1200, seed=42)
    )
    df.to_csv("data/raw/players.csv", index=False)

    # -------------------------
    # 2. Evaluation (Elbow + Silhouette)
    # -------------------------
    elbow_plot(
        df,
        k_min=2,
        k_max=10,
        outpath="reports/figures/elbow.png"
    )

    sil = silhouette(df, k=3)
    with open("reports/silhouette.txt", "w", encoding="utf-8") as f:
        f.write(f"Silhouette score (k=3): {sil:.4f}\n")

    # -------------------------
    # 3. Clustering
    # -------------------------
    model, scaler, labels = fit_kmeans(df, k=3, seed=42)
    summary = cluster_summary(df, labels)
    summary.to_csv("reports/cluster_summary.csv")

    # -------------------------
    # 4. Visualizations
    # -------------------------
    save_pca_scatter(
        df,
        labels,
        "reports/figures/pca_clusters.png"
    )

    save_cluster_profiles(
        summary,
        "reports/figures/cluster_profiles.png"
    )

    # -------------------------
    # Done
    # -------------------------
    print("✅ Pipeline finished successfully!")
    print("Generated files:")
    print("- data/raw/players.csv")
    print("- reports/cluster_summary.csv")
    print("- reports/figures/pca_clusters.png")
    print("- reports/figures/cluster_profiles.png")
    print("- reports/figures/elbow.png")
    print(f"- reports/silhouette.txt (k=3: {sil:.4f})")


if __name__ == "__main__":
    main()
