#!/usr/bin/env python
"""Visualize the labeled embeddings with UMAP and diagnostics."""

import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
EMBEDDINGS_PATH = PROJECT_ROOT / "data/labeled_frames_embeddings.pkl"
OUTPUT_DIR = PROJECT_ROOT / "data/camera_diagnostics"


def main():
    # Load embeddings with labels
    print("Loading embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    labels = data["labels"]
    paths = data["frame_paths"]

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Labels: {len([l for l in labels if l == 'main_angle'])} main, "
          f"{len([l for l in labels if l == 'other'])} other")

    # Convert labels to numeric
    y = np.array([0 if l == "main_angle" else 1 for l in labels])

    # Run UMAP
    print("\nRunning UMAP...")
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    # Compute diagnostics
    print("\nComputing diagnostics...")
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    sil = silhouette_score(embeddings, y, metric="cosine")
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    cv_scores = cross_val_score(knn, embeddings, y, cv=5)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Samples: {len(embeddings)} ({(y==0).sum()} main, {(y==1).sum()} other)")
    print(f"\nSilhouette Score: {sil:.3f}")
    if sil > 0.5:
        print("  -> Strong separation: threshold-based classification should work")
    elif sil > 0.25:
        print("  -> Moderate separation: may need tuning or simple classifier")
    else:
        print("  -> Weak separation: consider training a classifier")
    print(f"\nKNN Accuracy (5-fold CV): {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    print("=" * 60)

    # Create plot
    print("\nCreating visualization...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for label_val, color, name in [(0, "blue", "Main Angle"), (1, "red", "Other")]:
            mask = y == label_val
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=name,
                marker=dict(color=color, size=10, opacity=0.7),
                text=[Path(p).name for p, m in zip(paths, mask) if m],
                hoverinfo="text",
            ))

        fig.update_layout(
            title=f"Camera Angle Embeddings (Silhouette={sil:.3f}, KNN={cv_scores.mean():.1%})",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=1000,
            height=800,
        )

        html_path = OUTPUT_DIR / "labeled_embeddings_umap.html"
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")
        fig.show()

    except Exception as e:
        print(f"Plotly failed: {e}, using matplotlib")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        for label_val, color, name in [(0, "blue", "Main Angle"), (1, "red", "Other")]:
            mask = y == label_val
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name, alpha=0.7, s=50)
        ax.legend()
        ax.set_title(f"Camera Embeddings (Silhouette={sil:.3f})")
        plt.savefig(OUTPUT_DIR / "labeled_embeddings_umap.png", dpi=150)
        print(f"Saved: {OUTPUT_DIR / 'labeled_embeddings_umap.png'}")
        plt.show()


if __name__ == "__main__":
    main()
