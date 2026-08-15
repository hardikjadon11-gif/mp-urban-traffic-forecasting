"""
Graph construction utilities for traffic sensor networks.

Provides functions to build adjacency matrices and compute the
normalized graph Laplacian required by STGCN.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, eye as sparse_eye


def build_adjacency_from_coordinates(latitudes: np.ndarray, longitudes: np.ndarray,
                                     threshold_km: float = 5.0,
                                     sigma: float = None) -> np.ndarray:
    """
    Build a weighted adjacency matrix using Gaussian kernel on Haversine distances.

    Args:
        latitudes: Sensor latitudes in degrees.
        longitudes: Sensor longitudes in degrees.
        threshold_km: Maximum distance for an edge.
        sigma: Gaussian kernel bandwidth. Defaults to threshold_km / 2.

    Returns:
        adj: (N, N) weighted adjacency matrix.
    """
    n = len(latitudes)
    if sigma is None:
        sigma = threshold_km / 2

    lats = np.radians(latitudes)
    lons = np.radians(longitudes)

    # Haversine distance matrix
    lat_diff = lats[:, None] - lats[None, :]
    lon_diff = lons[:, None] - lons[None, :]
    a = np.sin(lat_diff / 2) ** 2 + \
        np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(lon_diff / 2) ** 2
    distances_km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # Gaussian kernel
    adj = np.exp(-(distances_km ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj, 0)
    adj[distances_km > threshold_km] = 0

    return adj


def build_adjacency_from_distance_matrix(distances: np.ndarray,
                                         threshold: float = None,
                                         sigma: float = 10.0) -> np.ndarray:
    """Build adjacency from a precomputed distance matrix."""
    if threshold is None:
        threshold = np.percentile(distances[distances > 0], 50)

    adj = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj, 0)
    adj[distances > threshold] = 0
    return adj


def compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}

    This is used by STGCN for graph convolution.
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # Degree vector

    # Handle isolated nodes (degree 0)
    d_inv_sqrt = np.zeros(n)
    nonzero = d > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])

    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    return L


def compute_scaled_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    Compute the scaled Laplacian: L_tilde = 2L/lambda_max - I

    Used for Chebyshev polynomial approximation in STGCN.
    """
    L = compute_normalized_laplacian(adj)
    n = L.shape[0]

    try:
        # Compute largest eigenvalue
        L_sparse = coo_matrix(L)
        eigenvalues = eigsh(L_sparse, k=1, which="LM", return_eigenvectors=False)
        lambda_max = eigenvalues[0]
    except Exception:
        lambda_max = 2.0  # Fallback for numerical issues

    if lambda_max < 1e-6:
        lambda_max = 2.0

    L_scaled = (2.0 * L / lambda_max) - np.eye(n)
    return L_scaled


def compute_cheb_polynomials(L_scaled: np.ndarray, K: int) -> list:
    """
    Compute Chebyshev polynomials T_0, T_1, ..., T_{K-1} of the scaled Laplacian.

    Used by STGCN for K-order graph convolution.

    Args:
        L_scaled: Scaled Laplacian matrix (N, N).
        K: Order of Chebyshev polynomial.

    Returns:
        List of K matrices, each (N, N).
    """
    n = L_scaled.shape[0]
    cheb = [np.eye(n)]  # T_0 = I

    if K > 1:
        cheb.append(L_scaled.copy())  # T_1 = L_scaled

    for k in range(2, K):
        # T_k = 2 * L_scaled * T_{k-1} - T_{k-2}
        cheb_k = 2 * L_scaled @ cheb[-1] - cheb[-2]
        cheb.append(cheb_k)

    return cheb


def get_graph_info(adj: np.ndarray) -> dict:
    """Get summary statistics about the graph."""
    n_nodes = adj.shape[0]
    n_edges = int((adj > 0).sum())
    avg_degree = n_edges / n_nodes if n_nodes > 0 else 0
    max_degree = int((adj > 0).sum(axis=1).max()) if n_nodes > 0 else 0
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round(avg_degree, 2),
        "max_degree": max_degree,
        "density": round(density, 4),
    }
