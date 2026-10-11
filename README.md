# PyTrajOpt

A lightweight, robust, and strongly typed **Python based Trajectory Optimization (PyTrajOpt)** framework built in Python using **CasADi** for symbolic automatic differentiation and **Pydantic** for runtime configuration validation.

---

## Overview

Solving Optimal Control Problems (OCPs) often requires writing boilerplate code to handle constraint dimension checks, bound mapping, and integrator setup. This repository provides a single-file core framework (`trajopt.py`) that abstracts these complexities while enforcing strict shape validation, automatic scaling, and clear separation between path and event constraints.

---

## 1. Theoretical Formulation (OCP)

The framework is designed to solve continuous-time Optimal Control Problems of the general form:

$$\min_{x(\cdot), u(\cdot), t_0, t_f} \quad J(x,u) = \int_{t_0}^{t_f} L(t,x,u)\mathrm{d}t + E(t_0,t_f,x_0,x_f)$$

Subject to the following system dynamics and constraints:

$$\dot{x}(t) = f(t,x,u), \quad t \in [t_0, t_f]$$

$$e_{lb} \leq e(t_0, t_f, x_0, x_f) \leq e_{ub}$$

$$h_{lb} \leq h(t,x,u) \leq h_{ub}$$

$$x_{lb} \leq x(t) \leq x_{ub}, \quad u_{lb} \leq u(t) \leq u_{ub}$$

### Variable & Function Definitions

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

---

## 2. Transcription: From OCP to NLP

To solve the infinite-dimensional continuous OCP on a digital computer, the problem is **transcribed** into a finite-dimensional **Nonlinear Program (NLP)** using direct collocation or multiple shooting.

### A. Time Grid Discretization

The continuous time horizon $[t_0, t_f]$ is divided into $n$ uniform intervals of length $\Delta t = \frac{t_f - t_0}{n}$, generating $n+1$ mesh grid points:

$$t_k = t_0 + k \Delta t, \quad k = 0, 1, \dots, n$$

### B. NLP Decision Vector ($W$) Structure

The continuous functions $x(t)$ and $u(t)$ are parameterized by state values at the grid nodes and control values over intervals. The NLP decision variable vector $W$ is assembled sequentially as follows:

$$W = \begin{bmatrix} t_0 & t_f & \text{vec}(X)^T & \text{vec}(U)^T \end{bmatrix}^T$$

* **Time variables:** Included only if free ($t_0$ and/or $t_f$).
* **State Matrix ($X$):** Dimensions $n_x \times (n+1)$, flattened into a column vector containing all states across all nodes:

$$X = \begin{bmatrix} x_0 & x_1 & \dots & x_n \end{bmatrix}$$


* **Control Matrix ($U$):** Dimensions $n_u \times n$, flattened into a column vector containing control inputs across all intervals:

$$U = \begin{bmatrix} u_0 & u_1 & \dots & u_{n-1} \end{bmatrix}$$



### C. Objective Function Discretization

The continuous integral (Lagrange term) is approximated using numerical quadrature (e.g., rectangular, trapezoidal, or Simpson integration), added to the terminal Mayer term:

$$J_{\text{NLP}}(W) = \frac{t_f - t_0}{n} \sum_{k=0}^{n-1} L(t_k, x_k, u_k) + E(t_0, t_f, x_0, x_n)$$

### D. Constraint Vector ($G(W)$) Structure

All problem constraints are stacked vertically into a single vector $G(W)$ bounded by lower and upper limits ($G_{lb} \leq G(W) \leq G_{ub}$):

1. **Event / Boundary Constraints:**

$$G_{\text{event}} = e(t_0, t_f, x_0, x_n), \quad e_{lb} \leq G_{\text{event}} \leq e_{ub}$$


2. **System Dynamics / Defect Constraints:**
Enforces physical consistency between nodes using the chosen integration method (e.g., $x_{k+1} - x_k - \Delta t f(\cdot) = 0$):

$$G_{\text{dyn}} = \text{Defects}(X, U, t) = 0$$


3. **Path Constraints:**
Evaluated across all grid nodes $k = 0, \dots, n$ (using zero-order hold extension for control inputs at the final node):

$$G_{\text{path}} = \begin{bmatrix} h(t_0, x_0, u_0) \\ \vdots \\ h(t_n, x_n, u_{n-1}) \end{bmatrix}, \quad \text{repmat}(h_{lb}, 1, n+1) \leq G_{\text{path}} \leq \text{repmat}(h_{ub}, 1, n+1)$$



---

## 3. Assembled Final NLP Formulation

Combining the objective function, stacked constraints, and decision vector bounds, the fully assembled **Nonlinear Program (NLP)** solved by CasADi and IPOPT takes the following standard mathematical form:

$$\min_{W} \quad J_{\text{NLP}}(W)$$

Subject to the composite constraints:

$$G_{lb} \leq G(W) \leq G_{ub}$$

where:

$$G(W) = \begin{bmatrix} e(t_0, t_f, x_0, x_n) \\ \text{Defects}(X, U, t) \\ h(t_0, x_0, u_0) \\ \vdots \\ h(t_n, x_n, u_{n-1}) \end{bmatrix}, \quad  G_{lb} = \begin{bmatrix} e_{lb} \\ \mathbf{0} \\ \text{repmat}(h_{lb}, 1, n+1) \end{bmatrix}, \quad  G_{ub} = \begin{bmatrix} e_{ub} \\ \mathbf{0} \\ \text{repmat}(h_{ub}, 1, n+1) \end{bmatrix}$$

And subject to simple variable bounds on the decision vector:

$$W_{lb} \leq W \leq W_{ub}$$

where:

$$W = \begin{bmatrix} t_0 \\ t_f \\ \text{vec}(X) \\ \text{vec}(U) \end{bmatrix}, \quad W_{lb} \begin{bmatrix} t_{0,lb} \\ t_{f,lb} \\ \text{repmat}(x_{lb}, 1, n+1) \\ \text{repmat}(u_{lb}, 1, n) \end{bmatrix}, \quad W_{ub} = \begin{bmatrix} t_{0,ub} \\ t_{f,ub} \\ \text{repmat}(x_{ub}, 1, n+1) \\ \text{repmat}(u_{ub}, 1, n) \end{bmatrix}$$

---

## 4. Discretization & Integration Methods

The framework supports multiple transcription strategies categorized into **Direct Multiple Shooting** and **Direct Collocation**:

* **Direct Multiple Shooting (Native & CasADi Integrators):**
* `fw_euler`: Forward Euler (explicit 1st order).
* `exp_mid`: Explicit Midpoint (2nd order Runge-Kutta).
* `rk4`: Runge-Kutta 4th order (via CasADi integrator).
* `cvodes`: Adaptive-step ODE/DAE integrator from SUNDIALS (ideal for stiff systems).
* `collocation`: Implicit collocation polynomials via CasADi.


* **Direct Collocation (Implicit Methods):**
* `bw_euler`: Backward Euler (implicit 1st order, highly stable).
* `trap`: Trapezoidal collocation (implicit 2nd order).
* `her_sim`: Hermite-Simpson collocation (cubic polynomial approximation).



---
### Post processing

Once solved the solution obtained can be visualized using the `plot_solution` function which creates a pdf with plots of the
optimal states, controls, costates and Hamiltonian.

For direct multiple shooting, estimates of the costates are extracted from the KKT multipliers, which is also used to 
compute the Hamiltonian over the grid. For the direct collocation methods, this is not implemented and will result in 
empty plots.

---

## 5. Repository Structure

```text
├── trajopt.py   # Core Pydantic configuration & CasADi NLP transcription engine
└── example.py   # Example script demonstrating setup, solving, and plotting

```

---

## 6. Installation

Ensure you have Python installed along with the required dependencies:

```bash
pip install casadi pydantic numpy matplotlib

```

---

## 7. Quick Start & Usage Example

Here is a brief example of how to configure and use `TrajOpt` in your script:

```python
import casadi as cs
from trajopt import casadi_integrator, plot_solution, TrajOpt


def low_thrust():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2

    # event and Mayer
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)

    # Lagrange dynamics and path
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = None

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", [t0s, tfs, x0s, xfs], [tfs])

    # event constraints
    e = cs.vertcat(
        x0s - cs.DM([1, 0, 0, 1]), xfs[0] - 4, xfs[2:, :] - cs.DM([0, 0.5]), tfs
    )
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0] * 7 + [0])
    ocp_data["e_ub"] = cs.vcat([0] * 7 + [100])

    # dynamics
    f = cs.vcat(
        [
            xs[2, :],
            xs[3, :] / xs[0, :],
            xs[3, :] ** 2 / xs[0, :] - 1 / xs[0, :] ** 2 + us[0, :],
            -xs[2, :] * xs[3, :] / xs[0, :] + us[1, :],
        ]
    )
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    print(ocp_data['f'].which_depends('i1',['o0']))

    method = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 4,
            "collocation_scheme": "legendre",
            "rootfinder":'kinsol',
            "number_of_finite_elements": 1,
            "simplify":True
        },
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )

    ocp_data["method"] = method

    ocp_data["u_lb"] = cs.DM([-0.01] * 2)
    ocp_data["u_ub"] = cs.DM([0.01] * 2)

    ocp_data["n"] = 100

    trajopt = TrajOpt(**ocp_data)

    nlp = trajopt.generate_NLP()
    nlp["nlp"].solver("ipopt")

    Xg = cs.vcat(
        [
            cs.linspace(1, 4, ocp_data["n"] + 1).T,
            cs.linspace(0, 0, ocp_data["n"] + 1).T,
            cs.linspace(0, 0, ocp_data["n"] + 1).T,
            cs.linspace(1, 0.5, ocp_data["n"] + 1).T,
        ]
    )

    nlp["nlp"].set_initial(nlp["X"], Xg)
    nlp["nlp"].set_initial(nlp["U"], cs.DM.zeros(nlp["U"].shape))
    nlp["nlp"].set_initial(nlp["tf"], 50)
    sol = nlp["nlp"].solve()
    plot_solution(sol, nlp)


if __name__ == "__main__":
    low_thrust()

```
