import numpy as np
from scipy.linalg import qr
def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return np.random.rand(n, m)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return x * a


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    return np.dot(x, y)


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.eye(dim)


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    if np.linalg.det(x) == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return np.linalg.inv(x)


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transosed matrix.
    """
    return x.T


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1th matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: hadamard produc
    """
    if x.shape != y.shape:
        raise ValueError("Matrices must have the same shape for Hadamard product.")
    return x * y


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    _, _, pivot_columns = qr(x, mode='economic', pivoting=True)
    return tuple(pivot_columns)


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    return np.linalg.norm(x, ord=order)


def test_matrix_operations():
    A = get_matrix(3, 3)
    B = get_matrix(3, 3)
    C = get_matrix(3, 1)
    
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Matrix C (Vector):\n", C)

    print("A + B:\n", add(A, B))
    print("A * 2:\n", scalar_multiplication(A, 2))
    print("A · C:\n", dot_product(A, C))
    print("Identity Matrix (3x3):\n", identity_matrix(3))
    
    try:
        print("Inverse of A:\n", matrix_inverse(A))
    except ValueError as e:
        print(e)
    
    print("Transpose of A:\n", matrix_transpose(A))
    print("Hadamard Product of A and B:\n", hadamard_product(A, B))
    print("Basis columns of A:\n", basis(A))
    print("Norm of A (Frobenius):", norm(A, 'fro'))

test_matrix_operations()
