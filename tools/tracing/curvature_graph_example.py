import numpy as np
import matplotlib.pyplot as plt

# 1. Define Control Points for the "Symmetrical Arch"
# P0, P1, P2, P3
pts = np.array([[0, 0], [2, 4], [8, 4], [10, 0]])


def get_bezier_data(pts, n=200):
  t = np.linspace(0, 1, n)

  # Bezier Curve Formula: (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
  x = (1-t)**3*pts[0, 0] + 3*(1-t)**2*t*pts[1, 0] + \
      3*(1-t)*t**2*pts[2, 0] + t**3*pts[3, 0]
  y = (1-t)**3*pts[0, 1] + 3*(1-t)**2*t*pts[1, 1] + \
      3*(1-t)*t**2*pts[2, 1] + t**3*pts[3, 1]

  # First Derivatives (x', y')
  dx = 3*(1-t)**2*(pts[1, 0]-pts[0, 0]) + 6*(1-t)*t * \
      (pts[2, 0]-pts[1, 0]) + 3*t**2*(pts[3, 0]-pts[2, 0])
  dy = 3*(1-t)**2*(pts[1, 1]-pts[0, 1]) + 6*(1-t)*t * \
      (pts[2, 1]-pts[1, 1]) + 3*t**2*(pts[3, 1]-pts[2, 1])

  # Second Derivatives (x'', y'')
  ddx = 6*(1-t)*(pts[2, 0]-2*pts[1, 0]+pts[0, 0]) + \
      6*t*(pts[3, 0]-2*pts[2, 0]+pts[1, 0])
  ddy = 6*(1-t)*(pts[2, 1]-2*pts[1, 1]+pts[0, 1]) + \
      6*t*(pts[3, 1]-2*pts[2, 1]+pts[1, 1])

  # Curvature Calculation
  curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5

  return t, x, y, curvature


t, x, y, kappa = get_bezier_data(pts)

# 2. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot the Curve
ax1.plot(x, y, 'b-', lw=3, label='Cubic Bezier')
ax1.plot(pts[:, 0], pts[:, 1], 'ro--', alpha=0.5, label='Control Polygon')
for i, p in enumerate(pts):
  ax1.text(p[0], p[1], f' P{i}', fontsize=12)
ax1.set_title("Bézier Curve (Symmetrical Arch)")
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True)

# Plot the Curvature
ax2.plot(t, kappa, 'g-', lw=2)
ax2.set_title("Curvature $\kappa(t)$")
ax2.set_xlabel("Parameter $t$ (0 to 1)")
ax2.set_ylabel("Curvature")
ax2.grid(True)

plt.tight_layout()
plt.show()
