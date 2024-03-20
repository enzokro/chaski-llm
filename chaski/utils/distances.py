import numpy as np
from numpy.typing import ArrayLike

# reference: https://github.com/chroma-core/chroma/blob/main/chromadb/utils/distance_functions.py

NORM_EPS = 1e-30  # Prevent division by zero

def l2_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the squared L2 distance between `x` and `y`."""
    return np.linalg.norm(x - y) ** 2

def inner_product(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the inner product between `x` and `y`."""
    return np.dot(x, y)

def cosine_similarity(x: ArrayLike, y: ArrayLike) -> float:
    """Finds the cosine similarity between `x` and `y`."""
    # flatten the vectors
    x_flat = np.array(x).squeeze()
    y_flat = np.array(y).squeeze()
    # find the dot product
    dot_product = inner_product(x_flat, y_flat)
    # vector norm for scaling
    norm_x = np.linalg.norm(x_flat) + NORM_EPS
    norm_y = np.linalg.norm(y_flat) + NORM_EPS
    # find and return the cosine similarity
    cos_sim =  dot_product / (norm_x * norm_y)
    return cos_sim

def cosine_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the cosine distance between vectors `x` and `y`."""
    return 1 - cosine_similarity(x, y)

def ip_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Calculates the inner product distance between `x` and `y`."""
    return 1 - np.dot(x, y)
