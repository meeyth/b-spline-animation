from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

# B-spline basis function (recursive Cox-de Boor)


def de_boor_cox(i, k, x, knots):
    """
    Recursively computes the B-spline basis function B(i, k) using the De Boor-Cox formula.
    Parameters:
    i : int Index of the basis function.
    k : int Degree of the B-spline.
    x : float The parameter value at which to evaluate.
    knot : np.ndarray The knot vector.
    Returns: float The computed basis function value B(i, k).
    """
    # Base case: B(i,1)
    if k == 1:
        return 1.0 if (knots[i] <= x < knots[i + 1]) else 0.0

    # Compute first term (left fraction)
    left_denom = knots[i + k - 1] - knots[i]
    left = ((x - knots[i]) / left_denom) * de_boor_cox(i,
                                                       k - 1, x, knots) if left_denom != 0 else 0

    # Compute second term (right fraction)
    right_denom = knots[i + k] - knots[i + 1]
    right = ((knots[i + k] - x) / right_denom) * \
        de_boor_cox(i + 1, k - 1, x, knots) if right_denom != 0 else 0

    res = left + right
    # print(f"N({i},{k}) = {res:.4f}")
    return res


def bspline_point(t, control_points, k, knots):
    n = len(control_points)
    point = np.zeros(2)
    for i in range(n):
        b = de_boor_cox(i, k, t, knots)
        point += b * control_points[i]
    return point


# Define control points; Pi
control_points = np.array([
    [0, 0],
    [1, 2],
    [2, 3],
    [4, 3.5],
    [5, 2]
])

n = len(control_points)
k = 4  # Cubic B-spline; (k - 1)
degree = k - 1  # Your k is order, so degree = k - 1


# Clamped knot vector
# knots = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])

knots = np.concatenate((
    np.zeros(k),
    np.linspace(0, 1, n - k + 1),
    np.ones(k)
))

# Parameter values where we want to evaluate the curve
t_vals = np.linspace(knots[k-1], knots[-k], 50)  # 20 points

# print(knots[degree], knots[-degree - 1])
# print(knots)
# print(t_vals)

# Compute and print curve points
for t in t_vals:
    point = bspline_point(t, control_points, k, knots)
    print(f"t = {t:.3f} --> point = ({point[0]:.4f}, {point[1]:.4f})")


# ------------------------------------------------------------------------------
#  ANIMATION SNIPPET
# Setup plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
curve_ax = ax[0]
basis_ax = ax[1]

curve_ax.set_title("B-spline Curve Construction")
basis_ax.set_title("Basis Functions")

curve_ax.plot(control_points[:, 0], control_points[:,
              1], 'ro--', label="Control Points")
curve_line, = curve_ax.plot([], [], 'b-', lw=2, label="B-spline Curve")
moving_point, = curve_ax.plot([], [], 'ko', label="Current Point")

colors = plt.cm.viridis(np.linspace(0, 1, n))
basis_lines = [basis_ax.plot([], [], color=colors[i], label=f"N{i},{degree}")[
    0] for i in range(n)]

curve_ax.legend()
basis_ax.set_xlim(knots[k - 1], knots[-k])
basis_ax.set_ylim(0, 1.1)
curve_ax.set_xlim(
    np.min(control_points[:, 0]) - 1, np.max(control_points[:, 0]) + 1)
curve_ax.set_ylim(
    np.min(control_points[:, 1]) - 1, np.max(control_points[:, 1]) + 1)
basis_ax.grid(True)
curve_ax.grid(True)

computed_points = []


def animate(frame):
    t = t_vals[frame]
    point = bspline_point(t, control_points, k, knots)
    computed_points.append(point)

    if len(computed_points) > 1:
        curve_line.set_data(*np.array(computed_points).T)
    moving_point.set_data([point[0]], [point[1]])

    # Update basis functions using de_boor_cox
    ts = np.linspace(knots[k - 1], knots[-k], 200)
    for i, line in enumerate(basis_lines):
        bs = [de_boor_cox(i, k, ti, knots) for ti in ts]
        line.set_data(ts, bs)

    return [curve_line, moving_point] + basis_lines


ani = FuncAnimation(fig, animate, frames=len(t_vals), interval=50, blit=True)
plt.tight_layout()
plt.show()
