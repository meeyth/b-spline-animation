import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# B-spline basis function (recursive Cox-de Boor)


def de_boor_basis(i, k, t, knots):
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]

    term1 = 0.0
    term2 = 0.0
    if denom1 > 0:
        term1 = ((t - knots[i]) / denom1) * de_boor_basis(i, k - 1, t, knots)
    if denom2 > 0:
        term2 = ((knots[i + k + 1] - t) / denom2) * \
            de_boor_basis(i + 1, k - 1, t, knots)

    return term1 + term2

# B-spline curve computation


def bspline_point(t, control_points, degree, knots):
    n = len(control_points)
    point = np.zeros(2)
    for i in range(n):
        b = de_boor_basis(i, degree, t, knots)
        point += b * control_points[i]
    return point


# Sample control points
control_points = np.array([
    [0, 0],
    [1, 2],
    [2, 3],
    [4, 3.5],
    [5, 2],
    [6, 0],
    [7, 2],
    [8, 4],
    [9, 4.5],
])
n = len(control_points)
degree = 3

# Clamped knot vector
knots = np.concatenate((
    np.zeros(degree),
    np.linspace(0, 1, n - degree + 1),
    np.ones(degree)
))

# t values for animation
t_vals = np.linspace(knots[degree], knots[-degree-1], 200)

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
basis_ax.set_xlim(knots[degree], knots[-degree-1])
basis_ax.set_ylim(0, 1.1)
curve_ax.set_xlim(-0.5, 9.5)
curve_ax.set_ylim(-1, 5)
basis_ax.grid(True)
curve_ax.grid(True)

computed_points = []


def animate(frame):
    t = t_vals[frame]
    point = bspline_point(t, control_points, degree, knots)
    computed_points.append(point)

    if len(computed_points) > 1:
        curve_line.set_data(*np.array(computed_points).T)

    moving_point.set_data([point[0]], [point[1]])

    # Update basis functions
    for i, line in enumerate(basis_lines):
        ts = np.linspace(knots[degree], knots[-degree-1], 200)
        bs = [de_boor_basis(i, degree, ti, knots) for ti in ts]
        line.set_data(ts, bs)

    return [curve_line, moving_point] + basis_lines


ani = FuncAnimation(fig, animate, frames=len(t_vals), interval=30, blit=True)
plt.tight_layout()
plt.show()
