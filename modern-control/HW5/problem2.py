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

# Parameters from the homework
MASS = 2.0
A_FRONT = 0.1
CD = 0.4
RHO = 1.204
MU = 0.2
G = 10.0

# Try multiple values to compare the energy/time tradeoff.
Q_VALUES = [0.0, 0.5, 3.0, 10.0]

# Energy used should not become negative during braking unless regenerative
# braking is explicitly modeled, so clip negative applied force to zero.
CLIP_NEGATIVE_FORCE = True

# Show combined plots comparing all q values.
SHOW_PLOTS = True


def nearest_velocity_index(v):
    return int(round(v / DV))


def solve_for_q(q):
    x_grid = np.arange(XI, XF + DX, DX)
    v_grid = np.arange(V_MIN, V_MAX + DV, DV)

    nx = len(x_grid)
    nv = len(v_grid)
    inf = 1e12

    cost_to_go = np.full((nx, nv), inf)
    policy_a = np.full((nx, nv), np.nan)
    policy_vnext = np.full((nx, nv), np.nan)
    policy_dt = np.full((nx, nv), np.nan)
    policy_dE = np.full((nx, nv), np.nan)
    policy_force = np.full((nx, nv), np.nan)

    # Terminal condition: only v = 0 at x = xf is feasible.
    for j, v in enumerate(v_grid):
        if np.isclose(v, 0.0):
            cost_to_go[-1, j] = 0.0

    for k in range(nx - 2, -1, -1):
        for j, v in enumerate(v_grid):
            best_cost = inf
            best_a = np.nan
            best_vn = np.nan
            best_dt = np.nan
            best_dE = np.nan
            best_force = np.nan

            for jn, v_next in enumerate(v_grid):
                if np.isclose(v, 0.0) and np.isclose(v_next, 0.0):
                    continue

                a = (v_next**2 - v**2) / (2 * DX)
                if a < A_MIN - 1e-9 or a > A_MAX + 1e-9:
                    continue

                if (v + v_next) <= 1e-12:
                    continue

                dt = 2 * DX / (v + v_next)

                # Over a constant-acceleration spatial step, v^2 varies linearly
                # with x, so the interval-average v^2 is the average of endpoints.
                v_sq_avg = 0.5 * (v**2 + v_next**2)
                drag_force = 0.5 * CD * RHO * A_FRONT * v_sq_avg
                force = MASS * a + drag_force + MU * MASS * G

                force_used = max(force, 0.0) if CLIP_NEGATIVE_FORCE else force
                dE = force_used * DX

                stage_cost = dE + q * dt
                total_cost = stage_cost + cost_to_go[k + 1, jn]

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_a = a
                    best_vn = v_next
                    best_dt = dt
                    best_dE = dE
                    best_force = force_used

            cost_to_go[k, j] = best_cost
            policy_a[k, j] = best_a
            policy_vnext[k, j] = best_vn
            policy_dt[k, j] = best_dt
            policy_dE[k, j] = best_dE
            policy_force[k, j] = best_force

    x_traj = [x_grid[0]]
    v_traj = [0.0]
    a_traj = []
    dt_traj = []
    dE_traj = []
    force_traj = []
    t_traj = [0.0]

    current_j = nearest_velocity_index(0.0)

    for k in range(nx - 1):
        a = policy_a[k, current_j]
        v_next = policy_vnext[k, current_j]
        dt = policy_dt[k, current_j]
        dE = policy_dE[k, current_j]
        force = policy_force[k, current_j]

        if np.isnan(a) or np.isnan(v_next):
            return None

        a_traj.append(a)
        dt_traj.append(dt)
        dE_traj.append(dE)
        force_traj.append(force)
        t_traj.append(t_traj[-1] + dt)
        x_traj.append(x_grid[k + 1])
        v_traj.append(v_next)

        current_j = nearest_velocity_index(v_next)

    total_time = float(np.sum(dt_traj))
    total_energy = float(np.sum(dE_traj))
    total_cost = total_energy + q * total_time

    return {
        "q": q,
        "x_traj": x_traj,
        "v_traj": v_traj,
        "a_traj": a_traj,
        "dt_traj": dt_traj,
        "dE_traj": dE_traj,
        "force_traj": force_traj,
        "t_traj": t_traj,
        "total_time": total_time,
        "total_energy": total_energy,
        "total_cost": total_cost,
    }


def print_solution(solution):
    print("===== ENERGY / TIME TRADEOFF SOLUTION =====")
    print(f"q = {solution['q']}")
    print(f"Total time   = {solution['total_time']:.4f} s")
    print(f"Total energy = {solution['total_energy']:.4f} J")
    print(f"Total cost J = {solution['total_cost']:.4f}")
    print()

    print("Step-by-step policy:")
    for k in range(len(solution["a_traj"])):
        print(
            f"x: {solution['x_traj'][k]:>4.1f} -> {solution['x_traj'][k + 1]:>4.1f} m, "
            f"v: {solution['v_traj'][k]:>4.1f} -> {solution['v_traj'][k + 1]:>4.1f} m/s, "
            f"a = {solution['a_traj'][k]:>6.3f} m/s^2, "
            f"F = {solution['force_traj'][k]:>6.3f} N, "
            f"dE = {solution['dE_traj'][k]:>6.3f} J, "
            f"dt = {solution['dt_traj'][k]:>6.3f} s"
        )
    print()

    if np.isclose(solution["q"], 0.0):
        print(
            "Note: with q = 0 the optimizer minimizes traction energy only, "
            "so it is willing to accept a very slow trip to reduce drag and "
            "acceleration effort."
        )
        print()


def plot_solutions(solutions):
    colors = [
        "#d62728",  # red
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]

    def plot_metric_figure(title, x_getter, y_getter, xlabel, ylabel):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, solution in enumerate(solutions):
            ax = axes[i]
            color = colors[i % len(colors)]
            xs = x_getter(solution)
            ys = y_getter(solution)
            ax.step(xs, ys, where="post", linewidth=2, color=color)
            ax.set_title(
                f"q = {solution['q']}, time = {solution['total_time']:.2f} s, "
                f"cost = {solution['total_cost']:.2f}"
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)

        for ax in axes[len(solutions):]:
            ax.axis("off")

        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_metric_figure(
        "Velocity Profiles",
        lambda solution: solution["x_traj"],
        lambda solution: solution["v_traj"],
        "Position x (m)",
        "Velocity v (m/s)",
    )
    plot_metric_figure(
        "Acceleration Profiles",
        lambda solution: solution["x_traj"][:-1],
        lambda solution: solution["a_traj"],
        "Position x (m)",
        "Acceleration a (m/s^2)",
    )
    plot_metric_figure(
        "Applied Force Profiles",
        lambda solution: solution["x_traj"][:-1],
        lambda solution: solution["force_traj"],
        "Position x (m)",
        "Applied Force F (N)",
    )
    plot_metric_figure(
        "Cumulative Energy Usage",
        lambda solution: solution["t_traj"],
        lambda solution: np.concatenate(([0.0], np.cumsum(solution["dE_traj"]))),
        "Time t (s)",
        "Cumulative Energy (J)",
    )

    plt.show()


def main():
    solutions = []

    for q in Q_VALUES:
        solution = solve_for_q(q)
        if solution is None:
            print(f"No feasible optimal path found from x=0, v=0 for q = {q}.")
            print()
            continue

        solutions.append(solution)
        print_solution(solution)

    if solutions:
        print("===== SUMMARY =====")
        for solution in solutions:
            print(
                f"q = {solution['q']:>4.1f} | "
                f"time = {solution['total_time']:>7.4f} s | "
                f"energy = {solution['total_energy']:>8.4f} J | "
                f"cost = {solution['total_cost']:>8.4f}"
            )

        print()
        print(
            "Interpretation: the q = 0 case minimizes energy alone, while larger "
            "q values increasingly trade extra energy for shorter trip time."
        )

        if SHOW_PLOTS:
            plot_solutions(solutions)


if __name__ == "__main__":
    main()

# Problem 2 shows the expected energy-time tradeoff: small q favors slow motion and low drag/acceleration effort; larger q favors faster travel at higher energy cost. However, the policy eventually converges to a terminal time, where there is diminishing return on further energy expenditure to reduce time.