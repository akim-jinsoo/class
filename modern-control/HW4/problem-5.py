import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
from scipy import signal

# Parameters
m = 1.0
R = 1.0
I = 1.0
k = 1.0

# Original state-space model
# x1 = y, x2 = y_dot
# (m + I/R^2) y_ddot + k y = F
A = np.array([
    [0.0, 1.0],
    [-k/(m + I/R**2), 0.0]
])

B = np.array([
    [0.0],
    [1.0/(m + I/R**2)]
])

C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

print("A =\n", A)
print("B =\n", B)
print("C =\n", C)
print("D =\n", D)

# Augmented system for LQI
# xI_dot = r - y
# xa = [x1, x2, xI]^T
Aa = np.block([
    [A, np.zeros((2,1))],
    [-C, np.zeros((1,1))]
])

Ba = np.vstack([B, [[0.0]]])
Er = np.array([[0.0], [0.0], [1.0]])   # reference input enters integrator
Ca = np.array([[1.0, 0.0, 0.0]])
Da = np.array([[0.0]])

# LQI design
def lqi(Aa, Ba, Q, R):
    P = solve_continuous_are(Aa, Ba, Q, np.array([[R]]))
    K = (1.0/R) * (Ba.T @ P)
    eigvals = np.linalg.eigvals(Aa - Ba @ K)
    return K, P, eigvals

Q1 = np.diag([1.0, 1.0, 10.0])
Q2 = np.diag([100.0, 1.0, 100.0])
Rval = 1.0

cases = {
    "Q1 = diag(1,1,10)": Q1,
    "Q2 = diag(100,1,100)": Q2
}

results = {}

for name, Q in cases.items():
    K, P, poles = lqi(Aa, Ba, Q, Rval)
    results[name] = {"K": K, "poles": poles}
    print(f"\n{name}")
    print("K =", K)
    print("Closed-loop poles =", poles)

# Simulation function
def simulate(K, r=1.0, t_final=15.0):
    Acl = Aa - Ba @ K

    def f(t, xa):
        return (Acl @ xa.reshape(-1,1) + Er * r).flatten()

    t_eval = np.linspace(0, t_final, 2000)
    x0 = np.array([0.0, 0.0, 0.0])   # zero initial conditions
    sol = solve_ivp(f, [0, t_final], x0, t_eval=t_eval)

    xa = sol.y
    y = xa[0, :]
    u = -(K @ xa).flatten()

    return sol.t, y, u

# Time-domain plot: output response
plt.figure(figsize=(8,5))
for name, data in results.items():
    t, y, u = simulate(data["K"])
    plt.plot(t, y, label=name)

plt.axhline(1.0, linestyle='--', linewidth=1, label='Reference y=1')
plt.xlabel("Time (s)")
plt.ylabel("Output y(t)")
plt.title("LQI closed-loop output response")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Time-domain plot: control effort
plt.figure(figsize=(8,5))
for name, data in results.items():
    t, y, u = simulate(data["K"])
    plt.plot(t, u, label=name)

plt.xlabel("Time (s)")
plt.ylabel("Control input u(t) = F(t)")
plt.title("LQI control effort")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Frequency-domain comparison
# Closed-loop from reference r to output y
# xa_dot = (Aa - BaK) xa + Er r
# y = Ca xa
plt.figure(figsize=(8,5))

for name, data in results.items():
    K = data["K"]
    Acl = Aa - Ba @ K

    sys_cl = signal.StateSpace(Acl, Er, Ca, Da)
    w, mag, phase = signal.bode(sys_cl)
    plt.semilogx(w, mag, label=name)

plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")
plt.title("Closed-loop Bode magnitude: reference to output")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()