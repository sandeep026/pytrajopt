# pytrajopt

A minimal package to solve optimal control problems in python via direct methods.

## Optimal control problem

pytrajopt currently solves the OCP of the form given below.

$$
J(x,u)=\int_{t_0}^{t_f}L(t,x,u)\mathrm{d}t+E(t_0,t_f,x_0,x_f)
$$ 

$$
e_{lb} \leq e(t_0,t_f,x_0,x_f) \leq e_{ub}
$$ 

$$
\dot{x}=f(t,x,u)
$$ 

$$
h_{lb} \leq h(t,x,u) \leq h_{ub}
$$

$$
x_{lb} \leq  x \leq x_{ub}
$$ 

$$
u_{lb} \leq  x \leq u_{ub}
$$

Here is the updated markdown table with separate definitions for both the initial state ($x_0$) and final state ($x_f$), alongside the time variables and other OCP components:

| Name | Description | Shape |
| --- | --- | --- |
| $t$ | Independent variable (Time) | $R$ |
| $t_0$ | Initial time | $R$ |
| $t_f$ | Final time | $R$ |
| $x(t)$ | State vector | $R^{n_x}$ |
| $x_0$ | Initial state | $R^{n_x}$ |
| $x_f$ | Final state | $R^{n_x}$ |
| $u(t)$ | Control input vector | $R^{n_u}$ |
| $J(x, u)$ | Objective function / Cost functional | $R$ |
| $L(t, x, u)$ | Running cost (Lagrange term) | $R$ |
| $E(t_0, t_f, x_0, x_f)$ | Terminal cost (Mayer term) | $R$ |
| $f(t, x, u)$ | System dynamics function | $R^{n_x}$ |
| $e(t_0, t_f, x_0, x_f)$ | Boundary / Endpoint constraint function | $R^{n_e}$ |
| $h(t, x, u)$ | Path constraint function | $R^{n_h}$ |
| $x_{lb}, x_{ub}$ | State lower and upper bounds | $R^{n_x}$ |
| $u_{lb}, u_{ub}$ | Control lower and upper bounds | $R^{n_u}$ |
| $e_{lb}, e_{ub}$ | Boundary constraint bounds | $R^{n_e}$ |
| $h_{lb}, h_{ub}$ | Path constraint bounds | $R^{n_h}$ |


## Numerical solution

### 1. Direct Multiple Shooting

In multiple shooting, the time horizon is divided into intervals. The states at the start of each interval are treated as decision variables, and an ODE/DAE integrator is used to propagate the dynamics across each interval while enforcing continuity constraints between them.

* **`fw_euler` (Forward Euler):** A native, explicit 1st-order integration scheme. Simple and fast, but requires small step sizes for stability and accuracy.
* **`exp_mid` (Explicit Midpoint):** A native 2nd-order Runge-Kutta method. Offers better accuracy than Forward Euler with moderate computational overhead.
* **`rk4` (Runge-Kutta 4th Order):** A standard high-accuracy 4th-order explicit integrator via CasADi. Widely used for smooth nonlinear systems.
* **`cvodes` (CVodes Integrator):** A powerful adaptive-step ODE/DAE integrator from the SUNDIALS suite wrapped in CasADi. Ideal for stiff systems and high-precision requirements.
* **`collocation` (CasADi Collocation Integrator):** Uses implicit collocation polynomials (like Radau or Legendre) within each multiple-shooting interval to handle difficult or stiff dynamics.

### 2. Direct Collocation

In direct collocation, the states and controls are parameterized over a mesh, and the continuous-time dynamics are approximated using implicit algebraic continuity equations at specific collocation points within each interval.

* **`bw_euler` (Backward Euler):** An implicit 1st-order method. Highly stable for stiff systems, though overly diffusive/low accuracy.
* **`trap` (Trapezoidal Method):** An implicit 2nd-order collocation method. Balances simplicity and accuracy well for a wide range of engineering problems.
* **`her_sim` (Hermite-Simpson):** A higher-order implicit collocation method (typically yielding 3rd or 4th-order accuracy) that uses cubic polynomial approximations over the intervals, providing an excellent trade-off between accuracy and sparse NLP structure.

The event constraints are evaluated at the boundaries and the path constraints are evaluated at the discrete grid points.
The integral in objective is approximated via a left Riemann sum.

If the OCP has free final or initial time, these are treated as free variables. Combining all the procedures leads to a sparse
nonlinear optimization problem.

## Optimization problem


