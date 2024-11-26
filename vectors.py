from typing import Sequence
import numpy as np
from scipy import sparse

#
def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """ 
    data = np.random.rand(dim // 2)
    rows = np.random.choice(dim, dim // 2, replace=False)
    return sparse.coo_matrix((data, (rows, np.zeros_like(rows))), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """ 
    return x * a


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    result = np.zeros_like(vectors[0])
    for vec, coeff in zip(vectors, coeffs):
        result += vec * coeff
    return result


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    return float(np.dot(x.T, y))


def norm(x: np.ndarray, order: int | float) -> float:
    """
    Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    return float(np.linalg.norm(x, ord=order))


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
     
    return float(np.linalg.norm(x - y))


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        np.ndarray: angle in deg.
    """
    dot = dot_product(x, y)
    norm_x = norm(x, 2)
    norm_y = norm(y, 2)
    cos_theta = dot / (norm_x * norm_y)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    return np.isclose(dot_product(x, y), 0.0)


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    return np.linalg.solve(a, b)



def test_functions():
    dim = 5
    x = get_vector(dim)
    y = get_vector(dim)

    print("Vector x:", x)
    print("Vector y:", y)

    sparse_x = get_sparse_vector(dim)
    print("Sparse Vector x:", sparse_x)

    print("Addition x + y:", add(x, y))
    print("Scalar Multiplication x * 2:", scalar_multiplication(x, 2))

    print("Dot Product x·y:", dot_product(x, y))
    print("Norm of x:", norm(x, 2))
    print("Distance between x and y:", distance(x, y))

    print("Cosine between x and y:", cos_between_vectors(x, y))
    print("Are x and y orthogonal:", is_orthogonal(x, y))

    vectors = [x, y]
    coeffs = [1.5, -0.5]
    print("Linear Combination of x and y:", linear_combination(vectors, coeffs))

    a = np.random.rand(dim, dim)
    b = np.random.rand(dim, 1)
    print("Linear System Solution:", solves_linear_systems(a, b))

test_functions()
