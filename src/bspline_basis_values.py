def de_boor_cox(i, k, x, knot):
    """
    Recursively computes the B-spline basis function B(i, k) using the De Boor-Cox formula.
    Parameters:
    i : int Index of the basis function.      k : int Degree of the B-spline.    x : float The parameter value at which to evaluate.       knot : list or array The knot vector.
    Returns: float The computed basis function value B(i, k).
    """
    # Base case: B(i,1)
    if k == 1:
        return 1.0 if (knot[i] <= x < knot[i + 1]) else 0.0

    # Compute first term (left fraction)
    left_denom = knot[i + k - 1] - knot[i]

    left = ((x - knot[i]) / left_denom) * de_boor_cox(i,
                                                      k - 1, x, knot) if left_denom != 0 else 0

    # Compute second term (right fraction)
    right_denom = knot[i + k] - knot[i + 1]

    right = ((knot[i + k] - x) / right_denom) * \
        de_boor_cox(i + 1, k - 1, x, knot) if right_denom != 0 else 0

    res = left + right

    print(f"B({i},{k}) = {res:.4f}")

    return res


def compute_b_spline_basis(x_values, knot, k):
    """
    Computes and prints all B-spline basis functions B(i, k) for multiple values of x.
    Parameters:   x_values : list or array The parameter values at which to evaluate.    knot : list or array The knot vector.   k : int The degree of the B-spline.
    Returns: dict Dictionary of computed basis function values.
    """
    n = len(knot) - k - 1  # Number of basis functions
    basis_functions = {i: [] for i in range(n)}
    print(f"\nB-spline Basis Functions for x in {x_values} (Degree {k}):\n")

    for x in x_values:
        print(f"x = {x:.1f}:")

        for i in range(n):
            value = de_boor_cox(i, k, x, knot)

            basis_functions[i].append(value)

            print(f"\tB({i},{k}) ({x}) = {value:.4f}")

        print()
        print("----------------------------------------------------------")
        print()

    return basis_functions


if __name__ == "__main__":
    knot = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]  # Knot vector

    k = 4  # Degree of the B-spline
    x_values = [1.8, 1.9, 2.0, 2.1, 2.2,]

    # Compute basis functions
    basis_functions = compute_b_spline_basis(x_values, knot, k)
