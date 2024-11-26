import numpy as np
from scipy.linalg import lu, qr, svd as scipy_svd

def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    P, L, U = lu(x)
    return P, L, U


def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    Q, R = qr(x)
    return Q, R


def determinant(x: np.ndarray) -> float:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        float: The determinant of the matrix.
    """
    return np.linalg.det(x)


def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    values, vectors = np.linalg.eig(x)
    return values, vectors


def svd_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    U, S, Vt = scipy_svd(x)
    return U, S, Vt



def test_decompositions():

    A = np.array([[4, 3], [6, 3]])

    P, L, U = lu_decomposition(A)
    print("LU Decomposition:")
    print("P:\n", P)
    print("L:\n", L)
    print("U:\n", U)

    Q, R = qr_decomposition(A)
    print("\nQR Decomposition:")
    print("Q:\n", Q)
    print("R:\n", R)

    det = determinant(A)
    print("\nDeterminant:")
    print("det(A):", det)

    eigenvalues, eigenvectors = eigen(A)
    print("\nEigenvalues and Eigenvectors:")
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    
    U, S, Vt = svd_decomposition(A)
    print("\nSingular Value Decomposition (SVD):")
    print("U:\n", U)
    print("S:\n", S)
    print("Vt:\n", Vt)



test_decompositions()
