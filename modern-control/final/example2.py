import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.linalg import expm, solve_discrete_lyapunov

# User-tunable discretization
DT = 0.5
T_STEPS = 9
DISCRETIZATION_METHOD = "zoh"   # options: "zoh", "euler", "tustin"

# Published example parameters
alpha = 1.0
Da = 20.0

beta_small = -0.45
beta_large = -1.00

Q = np.eye(2)
R = 0.01 * np.eye(2)
N = 2

# Initial state "x0 = I" is interpreted here as [1, 1]^T
x0 = np.ones(2)

# Soft constraint settings used for Fig. 9
xmax_soft = 2.0 * np.ones(2)
xmin_soft = -2.0 * np.ones(2)
M_soft = 2
T0_scalar = 10000.0
T_min_scalar = 10000.0
eps_min = 0.0


# Continuous and discrete models
def cstr_continuous_matrices(beta: float):
    """
    Continuous-time model from the paper:
      xdot = A x + B u
    with
      A = [[-(1+Da), -alpha*Da],
           [-beta*Da, -(1 + alpha*beta*Da)]]
      B = I
    """
    A = np.array([
        [-(1.0 + Da),       -alpha * Da],
        [-beta * Da, -(1.0 + alpha * beta * Da)]
    ], dtype=float)

    B = np.eye(2)
    return A, B


def zoh_discretize(Ac, Bc, dt):
    """
    Zero-order hold discretization using matrix exponential.
    """
    n = Ac.shape[0]
    m = Bc.shape[1]

    M = np.block([
        [Ac, Bc],
        [np.zeros((m, n)), np.zeros((m, m))]
    ])

    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:n + m]
    return Ad, Bd

def euler_discretize(Ac, Bc, dt):
    """
    Forward Euler discretization:
      x_{k+1} = (I + A dt) x_k + (B dt) u_k
    """
    n = Ac.shape[0]
    Ad = np.eye(n) + Ac * dt
    Bd = Bc * dt
    return Ad, Bd


def tustin_discretize(Ac, Bc, dt):
    """
    Tustin (bilinear) discretization:
      Ad = (I - A dt/2)^(-1) (I + A dt/2)
      Bd = (I - A dt/2)^(-1) B dt
    """
    n = Ac.shape[0]
    I = np.eye(n)

    inv_term = np.linalg.inv(I - 0.5 * Ac * dt)
    Ad = inv_term @ (I + 0.5 * Ac * dt)
    Bd = inv_term @ (Bc * dt)

    return Ad, Bd

def make_discrete_model(beta: float, dt: float):
    Ac, Bc = cstr_continuous_matrices(beta)

    if DISCRETIZATION_METHOD == "zoh":
        Ad, Bd = zoh_discretize(Ac, Bc, dt)

    elif DISCRETIZATION_METHOD == "euler":
        Ad, Bd = euler_discretize(Ac, Bc, dt)

    elif DISCRETIZATION_METHOD == "tustin":
        Ad, Bd = tustin_discretize(Ac, Bc, dt)

    else:
        raise ValueError(f"Unknown discretization method: {DISCRETIZATION_METHOD}")

    return Ad, Bd


# Cost helpers
def terminal_cost_matrix(A):
    """
    P solves:
      P = Q + A^T P A
    so that tail cost with zero future inputs is x^T P x.
    """
    return solve_discrete_lyapunov(A.T, Q)


def rollout_cost(A, B, x, u_seq):
    """
    Exact infinite-horizon cost used by the paper's formulation,
    represented as:
      sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
    where P solves the discrete Lyapunov equation.
    """
    P = terminal_cost_matrix(A)
    z = x.copy()
    cost = 0.0
    for uk in u_seq:
        cost += z @ Q @ z + uk @ R @ uk
        z = A @ z + B @ uk
    cost += z @ P @ z
    return cost


# Prediction matrices
def predict_states(A, B, x0, u_seq):
    """
    Returns [x1, x2, ..., xN]
    """
    xs = []
    x = x0.copy()
    for uk in u_seq:
        x = A @ x + B @ uk
        xs.append(x)
    return xs


# NMPC (nominal only)
def solve_nmpc(xk, A_nom, B_nom):
    """
    Nominal MPC with horizon N=2 and no explicit constraints.
    """
    u = cp.Variable((N, 2))

    x = xk
    cost = 0
    for k in range(N):
        cost += cp.quad_form(x, Q) + cp.quad_form(u[k], R)
        x = A_nom @ x + B_nom @ u[k]

    P_nom = terminal_cost_matrix(A_nom)
    cost += cp.quad_form(x, P_nom)

    problem = cp.Problem(cp.Minimize(cost))
    problem.solve(warm_start=True, verbose=False)

    if u.value is None:
        raise RuntimeError("NMPC failed")

    u_seq = np.array(u.value)
    return u_seq


# RMPC without soft state constraints (Figures 6 and 8 style)
def solve_rmpc_unconstrained(xk, A_nom, B_nom, model_set, p_hat):
    """
    Robust MPC:
      minimize nominal cost
      subject to cost non-increase constraints for each model
    """
    u = cp.Variable((N, 2))

    # nominal objective
    x = xk
    cost_nom = 0
    for k in range(N):
        cost_nom += cp.quad_form(x, Q) + cp.quad_form(u[k], R)
        x = A_nom @ x + B_nom @ u[k]
    P_nom = terminal_cost_matrix(A_nom)
    cost_nom += cp.quad_form(x, P_nom)

    constraints = []

    for A_i, B_i in model_set:
        # robust cost with decision sequence u
        x = xk
        cost_i = 0
        for k in range(N):
            cost_i += cp.quad_form(x, Q) + cp.quad_form(u[k], R)
            x = A_i @ x + B_i @ u[k]
        P_i = terminal_cost_matrix(A_i)
        cost_i += cp.quad_form(x, P_i)

        # feasible/reference cost with shifted previous sequence
        rhs = rollout_cost(A_i, B_i, xk, p_hat)

        constraints.append(cost_i <= rhs)

    problem = cp.Problem(cp.Minimize(cost_nom), constraints)
    problem.solve(warm_start=True, verbose=False)

    if u.value is None:
        raise RuntimeError("RMPC unconstrained failed")

    return np.array(u.value)



# Soft state-constraint helpers (Figure 9 style)
def build_soft_state_constraints(xs, eps, xmin, xmax):
    """
    For M prediction steps:
      xmin - e_j <= x_j <= xmax + e_j
      e_j >= eps_min
    eps is shape (M, n)
    """
    cons = []
    for j in range(M_soft):
        cons += [xs[j] <= xmax + eps[j]]
        cons += [xs[j] >= xmin - eps[j]]
        cons += [eps[j] >= eps_min]
    return cons


def solve_feasible_eps_for_shifted_input(xk, p_hat, model_set, eps_prev):
    """
    Compute a feasible slack sequence for the shifted input, as in the paper.
    Minimize distance to previous optimal slack.
    """
    eps = cp.Variable((M_soft, 2))

    objective = 0
    for j in range(M_soft):
        objective += cp.sum_squares(eps[j] - eps_prev[j])

    constraints = []
    for A_i, B_i in model_set:
        xs_i = predict_states(A_i, B_i, xk, p_hat[:M_soft])
        constraints += build_soft_state_constraints(xs_i, eps, xmin_soft, xmax_soft)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(warm_start=True, verbose=False)

    if eps.value is None:
        raise RuntimeError("Feasible slack QP failed")

    return np.array(eps.value)


def update_T_scalar(T_prev, eps_prev, eps_hat):
    """
    Paper recursion specialized to T_k = scalar * I.
    """
    num = 0.0
    den = 0.0
    for j in range(M_soft):
        num += eps_prev[j] @ (T_prev * np.eye(2)) @ eps_prev[j]
        den += eps_hat[j] @ (T_prev * np.eye(2)) @ eps_hat[j]

    den = max(den, 1e-12)
    return (num / den) * T_prev


def solve_rmpc_soft_constraints(xk, A_nom, B_nom, model_set, p_hat, eps_hat, T_scalar):
    """
    RMPC with soft state constraints, like Figure 9.
    """
    u = cp.Variable((N, 2))
    eps = cp.Variable((M_soft, 2))

    # nominal cost + slack penalty
    x = xk
    cost_nom = 0
    for k in range(N):
        cost_nom += cp.quad_form(x, Q) + cp.quad_form(u[k], R)
        x = A_nom @ x + B_nom @ u[k]
    P_nom = terminal_cost_matrix(A_nom)
    cost_nom += cp.quad_form(x, P_nom)

    for j in range(M_soft):
        cost_nom += T_scalar * cp.sum_squares(eps[j])

    constraints = []

    # soft state constraints enforced for all models
    for A_i, B_i in model_set:
        xs_i = predict_states(A_i, B_i, xk, u)
        constraints += build_soft_state_constraints(xs_i, eps, xmin_soft, xmax_soft)

    # robustness constraints for all models
    for A_i, B_i in model_set:
        x = xk
        cost_i = 0
        for k in range(N):
            cost_i += cp.quad_form(x, Q) + cp.quad_form(u[k], R)
            x = A_i @ x + B_i @ u[k]
        P_i = terminal_cost_matrix(A_i)
        cost_i += cp.quad_form(x, P_i)

        for j in range(M_soft):
            cost_i += T_scalar * cp.sum_squares(eps[j])

        rhs = rollout_cost(A_i, B_i, xk, p_hat)
        for j in range(M_soft):
            rhs += T_scalar * np.sum(eps_hat[j] ** 2)

        constraints.append(cost_i <= rhs)

    problem = cp.Problem(cp.Minimize(cost_nom), constraints)
    problem.solve(warm_start=True, verbose=False)

    if u.value is None or eps.value is None:
        raise RuntimeError("RMPC soft-constrained solve failed")

    return np.array(u.value), np.array(eps.value)



# Closed-loop simulators
def shift_sequence(u_seq):
    """
    p_hat = shifted previous optimal input sequence
    """
    p_hat = np.zeros_like(u_seq)
    p_hat[:-1] = u_seq[1:]
    p_hat[-1] = np.zeros(u_seq.shape[1])
    return p_hat


def simulate_nmpc(true_model, nominal_model):
    A_true, B_true = true_model
    A_nom, B_nom = nominal_model

    x = x0.copy()
    xs = [x.copy()]
    us = []

    # Compute one additional control action so input runs through time T_STEPS.
    for _ in range(T_STEPS + 1):
        u_seq = solve_nmpc(x, A_nom, B_nom)
        u0 = u_seq[0]
        us.append(u0.copy())
        x = A_true @ x + B_true @ u0
        xs.append(x.copy())

    return np.array(xs), np.array(us)


def simulate_rmpc(true_model, nominal_model, model_set):
    A_true, B_true = true_model
    A_nom, B_nom = nominal_model

    x = x0.copy()
    xs = [x.copy()]
    us = []

    p_prev = np.zeros((N, 2))

    # Compute one additional control action so input runs through time T_STEPS.
    for _ in range(T_STEPS + 1):
        p_hat = shift_sequence(p_prev)
        u_seq = solve_rmpc_unconstrained(x, A_nom, B_nom, model_set, p_hat)
        u0 = u_seq[0]
        us.append(u0.copy())
        x = A_true @ x + B_true @ u0
        xs.append(x.copy())
        p_prev = u_seq.copy()

    return np.array(xs), np.array(us)


def simulate_rmpc_soft(true_model, nominal_model, model_set):
    A_true, B_true = true_model
    A_nom, B_nom = nominal_model

    x = x0.copy()
    xs = [x.copy()]
    us = []
    eps_hist = []

    p_prev = np.zeros((N, 2))

    # Seed with a feasible slack profile for the shifted sequence.
    # Starting from eps_min can collapse T_scalar on the first update,
    # effectively removing soft-constraint pressure.
    eps_seed = eps_min * np.ones((M_soft, 2))
    eps_prev = solve_feasible_eps_for_shifted_input(x, p_prev, model_set, eps_seed)
    T_scalar = T0_scalar

    # Compute one additional control action so input runs through time T_STEPS.
    for _ in range(T_STEPS + 1):
        p_hat = shift_sequence(p_prev)
        eps_hat = solve_feasible_eps_for_shifted_input(x, p_hat, model_set, eps_prev)
        T_scalar = max(update_T_scalar(T_scalar, eps_prev, eps_hat), T_min_scalar)

        u_seq, eps_opt = solve_rmpc_soft_constraints(
            x, A_nom, B_nom, model_set, p_hat, eps_hat, T_scalar
        )

        u0 = u_seq[0]
        us.append(u0.copy())
        eps_hist.append(eps_opt.copy())

        x = A_true @ x + B_true @ u0
        xs.append(x.copy())

        p_prev = u_seq.copy()
        eps_prev = eps_opt.copy()

    return np.array(xs), np.array(us), np.array(eps_hist)



# Plotting
def plot_state_input(xs, us, title, y_lim=None, state_y_lim=None, input_y_lim=None):
    # Trim the final state so both traces share the same 1..(T_STEPS+1) time axis.
    n_plot = min(us.shape[0], xs.shape[0] - 1)
    t = np.arange(1, n_plot + 1)
    xs_plot = xs[:n_plot]
    us_plot = us[:n_plot]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    if y_lim is not None:
        if state_y_lim is None:
            state_y_lim = y_lim
        if input_y_lim is None:
            input_y_lim = y_lim

    axes[0].plot(t, xs_plot[:, 0], linewidth=2, label="x1")
    axes[0].plot(t, xs_plot[:, 1], linewidth=2, linestyle="--", label="x2")
    axes[0].axhline(0.0, linewidth=0.8, color="k")
    axes[0].set_ylabel("states")
    if state_y_lim is not None:
        axes[0].set_ylim(state_y_lim)
    y0_min, y0_max = axes[0].get_ylim()
    axes[0].set_yticks(np.linspace(y0_min, y0_max, 5))
    axes[0].set_xticks(t)
    axes[0].set_xlim(t[0], t[-1])
    axes[0].margins(x=0)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, us_plot[:, 0], linewidth=2, label="u1")
    axes[1].plot(t, us_plot[:, 1], linewidth=2, linestyle="--", label="u2")
    axes[1].axhline(0.0, linewidth=0.8, color="k")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("inputs")
    if input_y_lim is not None:
        axes[1].set_ylim(input_y_lim)
    y1_min, y1_max = axes[1].get_ylim()
    axes[1].set_yticks(np.linspace(y1_min, y1_max, 5))
    axes[1].set_xticks(t)
    axes[1].set_xlim(t[0], t[-1])
    axes[1].margins(x=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_state_input_with_soft_bounds(xs, us, title, y_lim=None):
    # Trim the final state so both traces share the same 1..(T_STEPS+1) time axis.
    n_plot = min(us.shape[0], xs.shape[0] - 1)
    t = np.arange(1, n_plot + 1)
    xs_plot = xs[:n_plot]
    us_plot = us[:n_plot]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    axes[0].plot(t, xs_plot[:, 0], linewidth=2, label="x1")
    axes[0].plot(t, xs_plot[:, 1], linewidth=2, linestyle="--", label="x2")
    axes[0].axhline(xmax_soft[0], linewidth=1.0, linestyle=":", color="k")
    axes[0].axhline(xmin_soft[0], linewidth=1.0, linestyle=":", color="k")
    axes[0].axhline(0.0, linewidth=0.8, color="k")
    axes[0].set_ylabel("states")
    if y_lim is not None:
        axes[0].set_ylim(y_lim)
    y0_min, y0_max = axes[0].get_ylim()
    axes[0].set_yticks(np.linspace(y0_min, y0_max, 5))
    axes[0].set_xticks(t)
    axes[0].set_xlim(t[0], t[-1])
    axes[0].margins(x=0)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, us_plot[:, 0], linewidth=2, label="u1")
    axes[1].plot(t, us_plot[:, 1], linewidth=2, linestyle="--", label="u2")
    axes[1].axhline(0.0, linewidth=0.8, color="k")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("inputs")
    if y_lim is not None:
        axes[1].set_ylim(y_lim)
    y1_min, y1_max = axes[1].get_ylim()
    axes[1].set_yticks(np.linspace(y1_min, y1_max, 5))
    axes[1].set_xticks(t)
    axes[1].set_xlim(t[0], t[-1])
    axes[1].margins(x=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes



# Main
if __name__ == "__main__":
    fig35_ylim = (-10, 10)

    model_small = make_discrete_model(beta_small, DT)
    model_large = make_discrete_model(beta_large, DT)
    model_set = [model_small, model_large]

    # Figure 5 style: NMPC on small-beta plant
    xs5, us5 = simulate_nmpc(true_model=model_small, nominal_model=model_small)
    fig5, _ = plot_state_input(
        xs5,
        us5,
        "Example 2: NMPC response for beta = -0.45",
        state_y_lim=(-1, 1),
        input_y_lim=(-4, 4),
    )

    # Figure 6 style: RMPC on small-beta plant
    xs6, us6 = simulate_rmpc(true_model=model_small, nominal_model=model_small, model_set=model_set)
    fig6, _ = plot_state_input(
        xs6,
        us6,
        "Example 2: RMPC response for beta = -0.45",
        state_y_lim=(-1, 1),
        input_y_lim=(-4, 4),
    )

    # Figure 7 style: NMPC on large-beta plant
    xs7, us7 = simulate_nmpc(true_model=model_large, nominal_model=model_small)
    fig7, _ = plot_state_input(xs7, us7, "Example 2: NMPC response for beta = -1.00", y_lim=fig35_ylim)

    # Figure 8 style: RMPC on large-beta plant
    xs8, us8 = simulate_rmpc(true_model=model_large, nominal_model=model_small, model_set=model_set)
    fig8, _ = plot_state_input(xs8, us8, "Example 2: RMPC response for beta = -1.00", y_lim=fig35_ylim)

    # Figure 9 style: RMPC on large-beta plant with soft state constraints
    xs9, us9, eps9 = simulate_rmpc_soft(true_model=model_large, nominal_model=model_small, model_set=model_set)
    
    print("Fig 8 max abs state:", np.max(np.abs(xs8), axis=0))
    print("Fig 9 max abs state:", np.max(np.abs(xs9), axis=0))
    print("Fig 9 max slack:", np.max(eps9, axis=(0, 1)))

    fig9, _ = plot_state_input_with_soft_bounds(
        xs9, us9,
        "Example 2: RMPC response for beta = -1.00 with soft state constraints",
        y_lim=fig35_ylim,
    )

    fig5.savefig("example2_figure5.pdf")
    fig6.savefig("example2_figure6.pdf")
    fig7.savefig("example2_figure7.pdf")
    fig8.savefig("example2_figure8.pdf")
    fig9.savefig("example2_figure9.pdf")

    plt.show()