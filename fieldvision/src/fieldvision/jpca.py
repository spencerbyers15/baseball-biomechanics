"""jPCA — rotational dynamics decomposition (Churchland et al., 2012).

Given a collection of trajectories X (T_k × N for each trial k), find the
skew-symmetric matrix M and basis where the dynamics ẋ = Mx best explains
the data. Skew-symmetric M has eigenvalues in conjugate pairs ±iω_k, each
corresponding to a 2D rotation plane oscillating at frequency ω_k.

Why this matters for preparatory waggle: PCA finds the directions of
maximum variance regardless of whether the dynamics in those directions
are rotational or just drifting. jPCA explicitly extracts the rotational
structure — the angle within the dominant rotation plane is a meaningful
phase of the oscillation, whereas the phase from Hilbert-transforming PC1
is only well-defined if PC1 happens to be one axis of a clean oscillation.
"""

from __future__ import annotations

import numpy as np


def fit_jpca(trajectories: list[np.ndarray], dt: float = 1.0):
    """Fit jPCA to a list of trajectories (each (T_k, N)).

    Parameters
    ----------
    trajectories : list of (T_k, N) arrays
        One trajectory per trial. Each trajectory should be mean-centered
        per-trial OR you should pre-subtract the cross-trial mean.
    dt : float
        Time step between consecutive samples (used to scale the angular
        frequency outputs into rad/s). Defaults to 1.0 (frequency in units
        of rad/sample).

    Returns
    -------
    dict with:
        M : (N, N) skew-symmetric matrix of best-fit rotational dynamics
        eigenvalues : (N,) complex eigenvalues of M (purely imaginary, in
            conjugate pairs, sorted by descending |imag|)
        eigenvectors : (N, N) complex eigenvectors of M, columns ordered
            to match eigenvalues
        rotation_planes : list of (e1, e2, omega) for each conjugate pair.
            e1, e2 are orthonormal real basis vectors of the plane (N,);
            omega is angular frequency in rad/{time unit of dt}.
    """
    X_chunks = []
    Xd_chunks = []
    for traj in trajectories:
        if len(traj) < 3:
            continue
        Xd = np.gradient(traj, dt, axis=0)
        X_chunks.append(traj)
        Xd_chunks.append(Xd)
    if not X_chunks:
        raise ValueError("no trajectories with length >= 3")
    X = np.vstack(X_chunks)
    Xd = np.vstack(Xd_chunks)

    X_mean = X.mean(axis=0)
    X = X - X_mean

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    M_unc = Xd.T @ X @ XtX_inv
    M = 0.5 * (M_unc - M_unc.T)

    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w.imag))
    w = w[order]
    V = V[:, order]

    rotation_planes = []
    used = set()
    for i, lam in enumerate(w):
        if i in used:
            continue
        if abs(lam.imag) < 1e-9:
            continue
        partner = None
        for j in range(i + 1, len(w)):
            if j in used: continue
            if abs(w[j] - np.conj(lam)) < 1e-6 * (abs(lam) + 1e-9):
                partner = j
                break
        if partner is None:
            continue
        used.add(i); used.add(partner)
        v = V[:, i]
        e1 = v.real.copy()
        e2 = v.imag.copy()
        n1 = np.linalg.norm(e1)
        if n1 < 1e-12: continue
        e1 = e1 / n1
        e2 = e2 - (e2 @ e1) * e1
        n2 = np.linalg.norm(e2)
        if n2 < 1e-12: continue
        e2 = e2 / n2
        omega = float(lam.imag)
        rotation_planes.append({
            "e1": e1, "e2": e2,
            "omega": omega,
            "freq_hz": omega / (2 * np.pi),
        })

    return {
        "M": M,
        "eigenvalues": w,
        "eigenvectors": V,
        "rotation_planes": rotation_planes,
        "X_mean": X_mean,
    }


def jpca_phase(X: np.ndarray, plane: dict, X_mean=None) -> np.ndarray:
    """Compute instantaneous angle for samples X within a given rotation plane.

    Parameters
    ----------
    X : (T, N) trajectory
    plane : dict with 'e1', 'e2' (from fit_jpca rotation_planes)
    X_mean : (N,) array, the mean used during fit_jpca (subtract before projecting)

    Returns
    -------
    phase : (T,) array of angles in [-π, π]
    """
    if X_mean is not None:
        Xc = X - X_mean
    else:
        Xc = X
    p1 = Xc @ plane["e1"]
    p2 = Xc @ plane["e2"]
    return np.arctan2(p2, p1)
