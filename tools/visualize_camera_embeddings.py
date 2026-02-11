#!/usr/bin/env python
"""Visualize camera angle embeddings with UMAP and diagnostic stats.

Usage:
    python tools/visualize_camera_embeddings.py [--labels PATH] [--embeddings PATH]

Requires:
    pip install umap-learn plotly scikit-learn kaleido
"""

import pickle
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data/reference_embeddings_sampled.pkl"
DEFAULT_LABELS = PROJECT_ROOT / "data/camera_angle_labels.json"
OUTPUT_DIR = PROJECT_ROOT / "data/camera_diagnostics"


def load_embeddings(embeddings_path: Path) -> tuple:
    """Load embeddings and frame paths."""
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["frame_paths"]


def load_labels(labels_path: Path) -> dict:
    """Load hand-labeled camera angles."""
    with open(labels_path, "r") as f:
        data = json.load(f)
    return data.get("labels", {})


def filter_to_labeled(embeddings: np.ndarray, frame_paths: list, labels: dict):
    """Filter embeddings to only include labeled frames."""
    # Normalize all label keys to forward slashes for matching
    normalized_labels = {k.replace("\\", "/"): v for k, v in labels.items()}

    mask = []
    filtered_labels = []
    filtered_paths = []

    for i, path in enumerate(frame_paths):
        # Normalize path for matching
        norm_path = path.replace("\\", "/")
        if norm_path in normalized_labels:
            mask.append(i)
            filtered_labels.append(normalized_labels[norm_path])
            filtered_paths.append(path)

    if not mask:
        return None, None, None

    filtered_embeddings = embeddings[mask]
    return filtered_embeddings, filtered_paths, filtered_labels


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    """Run UMAP dimensionality reduction."""
    try:
        import umap
    except ImportError:
        print("Installing umap-learn...")
        import subprocess
        subprocess.run(["pip", "install", "umap-learn"], check=True)
        import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def compute_diagnostics(embeddings: np.ndarray, labels: list) -> dict:
    """Compute cluster quality metrics."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    # Convert labels to numeric
    label_map = {"main_angle": 0, "other": 1}
    y = np.array([label_map.get(l, -1) for l in labels])

    # Filter out unknown labels
    valid = y >= 0
    X = embeddings[valid]
    y = y[valid]

    if len(np.unique(y)) < 2:
        return {
            "error": "Need at least 2 classes for diagnostics",
            "n_samples": len(y),
        }

    # Silhouette score (higher is better, -1 to 1)
    sil_score = silhouette_score(X, y, metric="cosine")

    # KNN cross-validation accuracy
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")

    # Compute centroids
    main_mask = y == 0
    other_mask = y == 1
    main_centroid = X[main_mask].mean(axis=0)
    other_centroid = X[other_mask].mean(axis=0)

    # Distance between centroids
    centroid_dist = np.linalg.norm(main_centroid - other_centroid)

    # Within-class variance
    main_var = np.var(np.linalg.norm(X[main_mask] - main_centroid, axis=1))
    other_var = np.var(np.linalg.norm(X[other_mask] - other_centroid, axis=1))

    # Compute distances from each sample to both centroids
    main_dists = np.linalg.norm(X - main_centroid, axis=1)
    other_dists = np.linalg.norm(X - other_centroid, axis=1)

    # Find optimal threshold (using centroid distance ratio)
    # threshold = dist_to_main / (dist_to_main + dist_to_other)
    ratios = main_dists / (main_dists + other_dists + 1e-8)

    # Find threshold that maximizes accuracy
    best_threshold = 0.5
    best_acc = 0
    for thresh in np.linspace(0.3, 0.7, 41):
        preds = (ratios > thresh).astype(int)
        acc = (preds == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    return {
        "n_samples": len(y),
        "n_main_angle": int(main_mask.sum()),
        "n_other": int(other_mask.sum()),
        "silhouette_score": float(sil_score),
        "silhouette_interpretation": interpret_silhouette(sil_score),
        "knn_cv_accuracy": float(cv_scores.mean()),
        "knn_cv_std": float(cv_scores.std()),
        "centroid_distance": float(centroid_dist),
        "main_angle_variance": float(main_var),
        "other_variance": float(other_var),
        "suggested_threshold": float(best_threshold),
        "threshold_accuracy": float(best_acc),
    }


def interpret_silhouette(score: float) -> str:
    """Interpret silhouette score."""
    if score > 0.5:
        return "Strong cluster separation - simple threshold should work"
    elif score > 0.25:
        return "Moderate separation - threshold may work with tuning"
    elif score > 0:
        return "Weak separation - consider training a classifier"
    else:
        return "No meaningful clusters - embedding space not separating classes"


def create_visualization(
    umap_coords: np.ndarray,
    labels: list,
    frame_paths: list,
    output_dir: Path,
):
    """Create interactive plotly visualization."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Installing plotly...")
        import subprocess
        subprocess.run(["pip", "install", "plotly", "kaleido"], check=True)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

    # Create color map
    colors = []
    for label in labels:
        if label == "main_angle":
            colors.append("blue")
        elif label == "other":
            colors.append("red")
        else:
            colors.append("gray")

    # Create hover text with frame info
    hover_texts = []
    for path, label in zip(frame_paths, labels):
        p = Path(path)
        stadium = p.parent.name
        hover_texts.append(f"Stadium: {stadium}<br>Frame: {p.name}<br>Label: {label}")

    # Create scatter plot
    fig = go.Figure()

    # Add points for each class
    for label_type, color, name in [
        ("main_angle", "blue", "Main Angle"),
        ("other", "red", "Other"),
    ]:
        mask = [l == label_type for l in labels]
        if not any(mask):
            continue

        coords = umap_coords[mask]
        texts = [t for t, m in zip(hover_texts, mask) if m]

        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            name=name,
            marker=dict(
                color=color,
                size=8,
                opacity=0.7,
            ),
            text=texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Camera Angle Embeddings (UMAP)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend=dict(x=0.02, y=0.98),
        hovermode="closest",
        width=1000,
        height=800,
    )

    # Save as HTML (interactive) and PNG
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "camera_embeddings_umap.html"
    fig.write_html(str(html_path))
    print(f"Saved interactive plot to: {html_path}")

    try:
        png_path = output_dir / "camera_embeddings_umap.png"
        fig.write_image(str(png_path))
        print(f"Saved static plot to: {png_path}")
    except Exception as e:
        print(f"Could not save PNG (kaleido may be missing): {e}")

    return fig


def create_matplotlib_backup(
    umap_coords: np.ndarray,
    labels: list,
    output_dir: Path,
):
    """Fallback matplotlib visualization."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    for label_type, color, name in [
        ("main_angle", "blue", "Main Angle"),
        ("other", "red", "Other"),
    ]:
        mask = np.array([l == label_type for l in labels])
        if not any(mask):
            continue
        ax.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=50,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Camera Angle Embeddings (UMAP)")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "camera_embeddings_umap_mpl.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved matplotlib plot to: {png_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize camera angle embeddings")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS,
                        help="Path to embeddings pickle file")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="Path to labels JSON file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Output directory for visualizations")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    args = parser.parse_args()

    # Check files exist
    if not args.embeddings.exists():
        print(f"Embeddings file not found: {args.embeddings}")
        print("Run build_full_reference.py first, or specify --embeddings")
        return

    if not args.labels.exists():
        print(f"Labels file not found: {args.labels}")
        print("Run label_camera_angles.py first to create labels")
        return

    print("Loading embeddings...")
    embeddings, frame_paths = load_embeddings(args.embeddings)
    print(f"Loaded {len(embeddings)} embeddings")

    print("Loading labels...")
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels")

    print("Filtering to labeled frames...")
    filtered_emb, filtered_paths, filtered_labels = filter_to_labeled(
        embeddings, frame_paths, labels
    )

    if filtered_emb is None or len(filtered_emb) == 0:
        print("No matching frames found between embeddings and labels!")
        print("Check that paths match between the two files.")
        return

    print(f"Found {len(filtered_emb)} labeled frames with embeddings")

    print("\nComputing UMAP projection...")
    umap_coords = compute_umap(filtered_emb, args.n_neighbors, args.min_dist)

    print("\nComputing diagnostics...")
    diagnostics = compute_diagnostics(filtered_emb, filtered_labels)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Total labeled samples: {diagnostics.get('n_samples', 'N/A')}")
    print(f"  - Main angle: {diagnostics.get('n_main_angle', 'N/A')}")
    print(f"  - Other: {diagnostics.get('n_other', 'N/A')}")
    print()
    print(f"Silhouette Score: {diagnostics.get('silhouette_score', 'N/A'):.3f}")
    print(f"  {diagnostics.get('silhouette_interpretation', '')}")
    print()
    print(f"KNN Cross-Validation Accuracy: {diagnostics.get('knn_cv_accuracy', 'N/A'):.1%} "
          f"(+/- {diagnostics.get('knn_cv_std', 0):.1%})")
    print()
    print(f"Centroid Distance: {diagnostics.get('centroid_distance', 'N/A'):.4f}")
    print(f"Main Angle Variance: {diagnostics.get('main_angle_variance', 'N/A'):.4f}")
    print(f"Other Variance: {diagnostics.get('other_variance', 'N/A'):.4f}")
    print()
    print(f"Suggested Threshold: {diagnostics.get('suggested_threshold', 'N/A'):.3f}")
    print(f"Threshold Accuracy: {diagnostics.get('threshold_accuracy', 'N/A'):.1%}")
    print("=" * 60)

    # Save diagnostics
    args.output.mkdir(parents=True, exist_ok=True)
    diag_path = args.output / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nSaved diagnostics to: {diag_path}")

    print("\nCreating visualization...")
    try:
        fig = create_visualization(umap_coords, filtered_labels, filtered_paths, args.output)
        # Show interactive plot
        fig.show()
    except Exception as e:
        print(f"Plotly visualization failed: {e}")
        print("Falling back to matplotlib...")
        create_matplotlib_backup(umap_coords, filtered_labels, args.output)

    print("\nDone! Check output in:", args.output)


if __name__ == "__main__":
    main()
