import numpy as np
from numpy.typing import ArrayLike

def l2(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the squared L2 distance between two vectors."""
    return np.sum((x - y) ** 2)

def cosine(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the cosine similarity between two vectors, adjusting for potential division by zero."""
    NORM_EPS = 1e-30  # Prevent division by zero
    x_flat = np.array(x).squeeze()
    y_flat = np.array(y).squeeze()
    dot_product = np.dot(x_flat, y_flat)
    norm_x = np.linalg.norm(x_flat) + NORM_EPS
    norm_y = np.linalg.norm(y_flat) + NORM_EPS
    dists = 1 - dot_product / (norm_x * norm_y)
    return dists

def ip(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the inner product similarity between two vectors."""
    return np.dot(x, y)
