import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Paper setup
a = 0.5
b_min = 4.0
b_max = 20.0
b_nom = b_min          # nominal model
q = 1.0
r = 0.001
x0 = 1.0

# Simulation length
T_steps = 10
DT_SECONDS = 1.0

# Plot settings
STATE_YLIM = (-1.0, 1.0)
INPUT_YLIM = (-0.2, 0.2)


# Cost and controller utilities

def stage_and_tail_cost(x_init, u_seq, a, b, q, r):
    """
    Infinite-horizon cost used in the paper's setup:
    - optimize first N inputs
    - then hold input at zero forever
    - since |a| < 1, tail cost is analytic

    J = sum_{k=0}^{N-1} (q z_k^2 + r u_k^2) + sum_{k=N}^{inf} q z_k^2
      = sum_{k=0}^{N-1} (q z_k^2 + r u_k^2) + q z_N^2 / (1 - a^2)
    """
    z = x_init
    cost = 0.0

    for uk in u_seq:
        cost += q * z**2 + r * uk**2
        z = a * z + b * uk

    cost += q * z**2 / (1.0 - a**2)
    return cost


def shifted_sequence(prev_u):
    """
    RLQR uses the shifted previous optimal input sequence:
      p_hat_k = [u_1*, u_2*, ..., u_{N-1}*, 0]
    """
    N = len(prev_u)
    if N == 1:
        return np.array([0.0])
    return np.hstack([prev_u[1:], 0.0])


def solve_nmpc(x, N, a, b_nom, q, r):
    """
    Nominal MPC:
      minimize nominal cost only
    """
    u = cp.Variable(N)

    z = x
    cost = 0
    for k in range(N):
        cost += q * z**2 + r * cp.square(u[k])
        z = a * z + b_nom * u[k]
    cost += q * z**2 / (1.0 - a**2)

    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if u.value is None:
        raise RuntimeError("NMPC solve failed.")

    return np.array(u.value).flatten()


def solve_rlqr(x, prev_u_opt, N, a, b_nom, b_min, b_max, q, r):
    """
    RLQR:
      minimize nominal cost
      subject to robustness constraints for b in {b_min, b_max}

    Constraint:
      U(x, u, b_i) <= U(x, p_hat, b_i)
    where p_hat is shifted previous optimal sequence.
    """
    u = cp.Variable(N)

    # Objective: nominal model cost
    z_nom = x
    nominal_cost = 0
    for k in range(N):
        nominal_cost += q * z_nom**2 + r * cp.square(u[k])
        z_nom = a * z_nom + b_nom * u[k]
    nominal_cost += q * z_nom**2 / (1.0 - a**2)

    u_hat = shifted_sequence(prev_u_opt)

    constraints = []
    for b in [b_min, b_max]:
        # Candidate sequence cost under plant b
        z_cand = x
        cand_cost = 0
        for k in range(N):
            cand_cost += q * z_cand**2 + r * cp.square(u[k])
            z_cand = a * z_cand + b * u[k]
        cand_cost += q * z_cand**2 / (1.0 - a**2)

        # Shifted feasible sequence cost under same plant b
        ref_cost = stage_and_tail_cost(x, u_hat, a, b, q, r)

        constraints.append(cand_cost <= ref_cost)

    prob = cp.Problem(cp.Minimize(nominal_cost), constraints)
    prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

    if u.value is None:
        raise RuntimeError("RLQR solve failed.")

    return np.array(u.value).flatten()


# Closed-loop simulation

def simulate_nmpc(true_b, N, steps, x0, a, b_nom, q, r):
    xs = [x0]
    us = []

    x = x0
    for _ in range(steps):
        u_seq = solve_nmpc(x, N, a, b_nom, q, r)
        u_apply = u_seq[0]

        us.append(u_apply)
        x = a * x + true_b * u_apply
        xs.append(x)

    return np.array(xs), np.array(us)


def simulate_rlqr(true_b, N, steps, x0, a, b_nom, b_min, b_max, q, r):
    xs = [x0]
    us = []

    x = x0
    prev_u_opt = np.zeros(N)

    for _ in range(steps):
        u_seq = solve_rlqr(x, prev_u_opt, N, a, b_nom, b_min, b_max, q, r)
        u_apply = u_seq[0]

        us.append(u_apply)
        x = a * x + true_b * u_apply
        xs.append(x)

        prev_u_opt = u_seq.copy()

    return np.array(xs), np.array(us)


# Plotting

def plot_nmpc_vs_rlqr(g, N=1):
    true_b = g * b_nom

    xs_nmpc, us_nmpc = simulate_nmpc(
        true_b=true_b, N=N, steps=T_steps, x0=x0,
        a=a, b_nom=b_nom, q=q, r=r
    )

    xs_rlqr, us_rlqr = simulate_rlqr(
        true_b=true_b, N=N, steps=T_steps, x0=x0,
        a=a, b_nom=b_nom, b_min=b_min, b_max=b_max, q=q, r=r
    )

    n_plot = min(len(us_nmpc), len(xs_nmpc) - 1)
    t = np.arange(1, n_plot + 1) * DT_SECONDS
    xs_nmpc_plot = xs_nmpc[:n_plot]
    xs_rlqr_plot = xs_rlqr[:n_plot]
    us_nmpc_plot = us_nmpc[:n_plot]
    us_rlqr_plot = us_rlqr[:n_plot]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)

    # State
    axes[0].plot(t, xs_nmpc_plot, label="NMPC", linewidth=2)
    axes[0].plot(t, xs_rlqr_plot, "--", label="RLQR", linewidth=2)
    axes[0].axhline(0.0, color="k", linewidth=0.8)
    axes[0].set_ylabel("State")
    axes[0].set_ylim(STATE_YLIM)
    axes[0].set_yticks(np.linspace(STATE_YLIM[0], STATE_YLIM[1], 5))
    axes[0].set_xticks(t)
    axes[0].set_xlim(t[0], t[-1])
    axes[0].margins(x=0)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].set_title(f"Example 1: NMPC and RLQR responses for g = {g}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Input
    axes[1].plot(t, us_nmpc_plot, label="NMPC", linewidth=2)
    axes[1].plot(t, us_rlqr_plot, "--", label="RLQR", linewidth=2)
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Input")
    axes[1].set_ylim(INPUT_YLIM)
    axes[1].set_yticks(np.linspace(INPUT_YLIM[0], INPUT_YLIM[1], 5))
    axes[1].set_xticks(t)
    axes[1].set_xlim(t[0], t[-1])
    axes[1].margins(x=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_rlqr_horizon_study(g=5, horizons=(1, 5, 10)):
    true_b = g * b_nom

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=False)
    line_styles = ["-", "--", ":", "-."]
    line_width = 2.0
    center_idx = (len(horizons) - 1) / 2.0
    state_offset_step = 0.01 * (STATE_YLIM[1] - STATE_YLIM[0])
    input_offset_step = 0.01 * (INPUT_YLIM[1] - INPUT_YLIM[0])

    for idx, N in enumerate(horizons):
        xs, us = simulate_rlqr(
            true_b=true_b, N=N, steps=T_steps, x0=x0,
            a=a, b_nom=b_nom, b_min=b_min, b_max=b_max, q=q, r=r
        )

        n_plot = min(len(us), len(xs) - 1)
        t = np.arange(1, n_plot + 1) * DT_SECONDS

        # Curves can overlap exactly; apply a small vertical visual offset per horizon.
        offset_idx = idx - center_idx
        xs_plot = xs[:n_plot] + offset_idx * state_offset_step
        us_plot = us[:n_plot] + offset_idx * input_offset_step

        axes[0].plot(
            t,
            xs_plot,
            linewidth=line_width,
            linestyle=line_styles[idx % len(line_styles)],
            label=f"N = {N}",
        )
        axes[1].plot(
            t,
            us_plot,
            linewidth=line_width,
            linestyle=line_styles[idx % len(line_styles)],
            label=f"N = {N}",
        )

    axes[0].axhline(0.0, color="k", linewidth=0.8)
    axes[0].set_ylabel("State")
    axes[0].set_ylim(STATE_YLIM)
    axes[0].set_yticks(np.linspace(STATE_YLIM[0], STATE_YLIM[1], 5))
    axes[0].set_xticks(t)
    axes[0].set_xlim(t[0], t[-1])
    axes[0].margins(x=0)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].set_title("Example 1: RLQR responses for N = 1, 5, 10 with g = 5")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.0, color="k", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Input")
    axes[1].set_ylim(INPUT_YLIM)
    axes[1].set_yticks(np.linspace(INPUT_YLIM[0], INPUT_YLIM[1], 5))
    axes[1].set_xticks(t)
    axes[1].set_xlim(t[0], t[-1])
    axes[1].margins(x=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# Main

if __name__ == "__main__":
    # Figures 1-3 style
    fig1, _ = plot_nmpc_vs_rlqr(g=5, N=1)
    fig2, _ = plot_nmpc_vs_rlqr(g=3, N=1)
    fig3, _ = plot_nmpc_vs_rlqr(g=1, N=1)

    # Figure 4 style
    fig4, _ = plot_rlqr_horizon_study(g=5, horizons=(1, 5, 10))

    fig1.savefig("example1_figure1.pdf")
    fig2.savefig("example1_figure2.pdf")
    fig3.savefig("example1_figure3.pdf")
    fig4.savefig("example1_figure4.pdf")

    plt.show()