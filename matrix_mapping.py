import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return -x


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    if x.ndim == 1:  
        return x[::-1]
    elif x.ndim == 2:  
        return x[::-1, ::-1]
    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """
    Compute affine transformation.

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in degrees.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    alpha_rad = np.radians(alpha_deg)
    
    S = np.array([[scale[0], 0], [0, scale[1]]])

    R = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                  [np.sin(alpha_rad),  np.cos(alpha_rad)]])

    Sh = np.array([[1, shear[0]], [shear[1], 1]])

    T = S @ R @ Sh

    transformed = x @ T.T

    transformed += np.array(translate)

    return transformed


def test_transformations():
    
    vector = np.array([[1], [2], [3]])
    matrix = np.array([[1, 2], [3, 4]])

    print("Original Vector:\n", vector)
    print("Negated Vector:\n", negative_matrix(vector))
    print("Reversed Vector:\n", reverse_matrix(vector))

    print("Original Matrix:\n", matrix)
    print("Negated Matrix:\n", negative_matrix(matrix))
    print("Reversed Matrix:\n", reverse_matrix(matrix))

    matrix_to_transform = np.array([[1, 0], [0, 1]])
    transformed_matrix = affine_transform(
        matrix_to_transform,
        alpha_deg=45,
        scale=(1, 1),
        shear=(0.5, 0.5),
        translate=(2, 3)
    )
    print("Affine Transformed Matrix:\n", transformed_matrix)



test_transformations()
