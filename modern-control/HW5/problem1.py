import numpy as np
import matplotlib.pyplot as plt


DX = 1.0
DV = 0.5
XI = 0.0
XF = 10.0
V_MIN = 0.0
V_MAX = 5.0
A_MIN = -3.0
A_MAX = 3.0

# Discretized grids
x_grid = np.arange(XI, XF + DX, DX)
v_grid = np.arange(V_MIN, V_MAX + DV, DV)

nx = len(x_grid)
nv = len(v_grid)

INF = 1e12

# Cost-to-go and policy tables
J = np.full((nx, nv), INF)
policy_a = np.full((nx, nv), np.nan)
policy_vnext = np.full((nx, nv), np.nan)
policy_dt = np.full((nx, nv), np.nan)

# Terminal condition: only v = 0 at x = xf is feasible
for j, v in enumerate(v_grid):
    if np.isclose(v, 0.0):
        J[-1, j] = 0.0


def nearest_velocity_index(v):
    return int(round(v / DV))


# Backward DP
for k in range(nx - 2, -1, -1):
    for j, v in enumerate(v_grid):
        best_cost = INF
        best_a = np.nan
        best_vn = np.nan
        best_dt_local = np.nan

        # Try all reachable next velocities on the velocity grid
        for jn, v_next in enumerate(v_grid):
            # Avoid staying at zero speed while trying to move forward
            if np.isclose(v, 0.0) and np.isclose(v_next, 0.0):
                continue

            # From kinematics over one spatial step
            a = (v_next**2 - v**2) / (2 * DX)

            # Check acceleration bounds
            if a < A_MIN - 1e-9 or a > A_MAX + 1e-9:
                continue

            # Need positive forward motion over the interval
            if (v + v_next) <= 1e-12:
                continue

            # Time for this step
            dt = 2 * DX / (v + v_next)

            # Total cost
            cost = dt + J[k + 1, jn]

            if cost < best_cost:
                best_cost = cost
                best_a = a
                best_vn = v_next
                best_dt_local = dt

        J[k, j] = best_cost
        policy_a[k, j] = best_a
        policy_vnext[k, j] = best_vn
        policy_dt[k, j] = best_dt_local

# Reconstruct optimal trajectory starting from x=0, v=0
x_traj = [x_grid[0]]
v_traj = [0.0]
a_traj = []
dt_traj = []
t_traj = [0.0]

current_v = 0.0
current_j = nearest_velocity_index(current_v)

feasible = True

for k in range(nx - 1):
    a = policy_a[k, current_j]
    v_next = policy_vnext[k, current_j]
    dt = policy_dt[k, current_j]

    if np.isnan(a) or np.isnan(v_next):
        feasible = False
        break

    a_traj.append(a)
    dt_traj.append(dt)
    t_traj.append(t_traj[-1] + dt)
    x_traj.append(x_grid[k + 1])
    v_traj.append(v_next)

    current_j = nearest_velocity_index(v_next)

if not feasible:
    print("No feasible optimal path found from x=0, v=0.")
else:
    print("===== MINIMUM-TIME SOLUTION =====")
    print(f"Optimal total time = {J[0, nearest_velocity_index(0.0)]:.4f} s")
    print()

    print("Step-by-step policy:")
    for k in range(nx - 1):
        print(
            f"x: {x_traj[k]:>4.1f} -> {x_traj[k + 1]:>4.1f} m, "
            f"v: {v_traj[k]:>4.1f} -> {v_traj[k + 1]:>4.1f} m/s, "
            f"a = {a_traj[k]:>6.3f} m/s^2, "
            f"dt = {dt_traj[k]:>6.3f} s"
        )

    # Plot velocity vs position
    plt.figure(figsize=(8, 4))
    plt.step(x_traj, v_traj, where="post")
    plt.xlabel("Position x (m)")
    plt.ylabel("Velocity v (m/s)")
    plt.title("Optimal Velocity Profile (Minimum Time)")
    plt.grid(True)
    plt.tight_layout()

    # Plot acceleration vs position interval
    plt.figure(figsize=(8, 4))
    plt.step(x_traj[:-1], a_traj, where="post")
    plt.xlabel("Position x (m)")
    plt.ylabel("Acceleration a (m/s^2)")
    plt.title("Optimal Acceleration Profile (Minimum Time)")
    plt.grid(True)
    plt.tight_layout()

    # Plot velocity vs time
    plt.figure(figsize=(8, 4))
    plt.step(t_traj, v_traj, where="post")
    plt.xlabel("Time t (s)")
    plt.ylabel("Velocity v (m/s)")
    plt.title("Velocity vs Time (Minimum Time)")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

# Problem 1 optimal policy accelerates to near the speed limit and then symmetrically decelerates.