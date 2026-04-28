import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# System parameters
R1 = 1.0
R2 = 3.0
C1 = 0.05
C2 = 0.025
R3 = 5.0
L  = 1.0

# State variables:
# x = [q1, q2, lam]^T
# q1 = charge on C1
# q2 = charge on C2
# lam = flux linkage of L
#
# v_C1 = q1/C1
# v_C2 = q2/C2
# i_L  = lam/L

A = np.array([
    [-1.0/(C1*(R1+R2)),  0.0,          -1.0/L],
    [0.0,                0.0,           1.0/L],
    [1.0/C1,            -1.0/C2,      -R3/L]
], dtype=float)

B = np.array([
    [1.0/(R1+R2)],
    [0.0],
    [0.0]
], dtype=float)

C = np.array([[1.0/C1, 0.0, 0.0]], dtype=float)
D = np.array([[0.0]], dtype=float)

print("A =\n", A)
print("B =\n", B)
print("C =\n", C)
print("D =\n", D)

# Stability
open_loop_poles = eigvals(A)
print("\nOpen-loop poles:")
print(open_loop_poles)

# Controllability
Ctrb = np.hstack([B, A @ B, A @ A @ B])
print("\nControllability matrix =\n", Ctrb)
print("Rank of controllability matrix =", np.linalg.matrix_rank(Ctrb))

# LQR function
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    eig_cl = eigvals(A - B @ K)
    return K, P, eig_cl

# Closed-loop simulation
def simulate_closed_loop(K, x0, t_end=5.0):
    Acl = A - B @ K

    def f(t, x):
        return Acl @ x

    t_eval = np.linspace(0, t_end, 1000)
    sol = solve_ivp(f, [0, t_end], x0, t_eval=t_eval)

    x = sol.y
    u = -(K @ x).flatten()
    y = (C @ x).flatten()   # output voltage across C1

    return sol.t, x, y, u

# Initial condition:
# 0.01 charge left in C1, C2 and L initially at rest
x0 = np.array([0.01, 0.0, 0.0])

# Four LQR cases
cases = {
    "Q=I, R=0.1": (
        np.diag([1.0, 1.0, 1.0]),
        np.array([[0.1]])
    ),
    "Q=I, R=10": (
        np.diag([1.0, 1.0, 1.0]),
        np.array([[10.0]])
    ),
    "R=1, q1 weight=1": (
        np.diag([1.0, 1.0, 1.0]),
        np.array([[1.0]])
    ),
    "R=1, q1 weight=100": (
        np.diag([100.0, 1.0, 1.0]),
        np.array([[1.0]])
    )
}

results = {}

for name, (Q, R) in cases.items():
    K, P, poles = lqr(A, B, Q, R)
    results[name] = {"K": K, "P": P, "poles": poles}

    print(f"\n{name}")
    print("Q =\n", Q)
    print("R =\n", R)
    print("K =", K)
    print("Closed-loop poles =", poles)

# Plot output y(t) = v_C1(t)
plt.figure(figsize=(8, 5))
for name, data in results.items():
    K = data["K"]
    t, x, y, u = simulate_closed_loop(K, x0)
    plt.plot(t, y, label=name)

plt.xlabel("Time (s)")
plt.ylabel("y(t) = v_C1(t) [V]")
plt.title("Closed-loop output response using states [q1, q2, lambda]^T")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot control effort u(t) = E(t)
plt.figure(figsize=(8, 5))
for name, data in results.items():
    K = data["K"]
    t, x, y, u = simulate_closed_loop(K, x0)
    plt.plot(t, u, label=name)

plt.xlabel("Time (s)")
plt.ylabel("u(t) = E(t) [V]")
plt.title("Closed-loop control effort using states [q1, q2, lambda]^T")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()