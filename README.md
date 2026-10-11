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

The problem is taken from "Ross, I. Michael, Qi Gong, and Pooya Sekhavat. "Low-thrust, high-accuracy trajectory optimization." Journal of Guidance, Control, and Dynamics 30.4 (2007): 921-933". The minimum time reported when solved using LGL pseudospectral
method is **47.706** for 300 nodes.

The primal and dual solution for the problem are given in [here](./solution.pdf)

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

Output

```shell
******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.2.

Number of nonzeros in equality constraint Jacobian...:     2907
Number of nonzeros in inequality constraint Jacobian.:      606
Number of nonzeros in Lagrangian Hessian.............:     2595

Total number of variables............................:      605
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:      407
Total number of inequality constraints...............:      606
        inequality constraints with only lower bounds:        1
   inequality constraints with lower and upper bounds:      201
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  5.0000000e+01 5.00e-01 0.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.8418068e+01 1.53e-01 4.51e-01  -1.0 1.68e+01    -  7.43e-01 6.93e-01f  1
   2  3.7679442e+01 1.14e-01 1.99e+01  -1.0 5.27e+00    -  5.56e-01 2.57e-01h  1
   3  4.1896931e+01 7.82e-02 3.62e+01  -1.0 1.34e+01    -  7.36e-01 3.14e-01h  1
   4  4.5322820e+01 5.34e-02 8.92e+01  -1.0 1.08e+01    -  1.00e+00 3.18e-01h  1
   5  4.9737276e+01 3.63e-02 5.96e+01  -1.0 1.39e+01    -  2.80e-01 3.18e-01h  1
   6  5.3456566e+01 2.47e-02 5.56e+01  -1.0 1.16e+01    -  2.68e-01 3.20e-01h  1
   7  5.5843323e+01 1.16e-02 4.26e+01  -1.0 4.48e+00    -  3.23e-01 5.32e-01h  1
   8  5.6501409e+01 8.89e-03 3.00e+01  -1.0 2.79e+00    -  5.31e-01 2.36e-01f  1
   9  5.7226895e+01 7.30e-03 2.59e+01  -1.0 4.06e+00    -  2.53e-01 1.79e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  5.8116214e+01 4.05e-03 2.97e+01  -1.0 2.01e+00    -  1.00e+00 4.43e-01h  1
  11  5.8754833e+01 1.25e-04 3.46e-01  -1.0 6.39e-01    -  1.00e+00 1.00e+00h  1
  12  5.5128078e+01 3.40e-03 1.04e+01  -1.7 8.39e+00    -  3.17e-01 4.32e-01f  1
  13  5.3215965e+01 1.49e-03 1.29e+01  -1.7 3.31e+00    -  4.49e-01 5.78e-01f  1
  14  5.2114227e+01 4.30e-04 8.99e+00  -1.7 1.58e+00    -  5.70e-01 6.95e-01f  1
  15  5.1551682e+01 2.43e-04 2.68e+00  -1.7 9.35e-01    -  6.73e-01 6.02e-01f  1
  16  5.1308148e+01 2.76e-04 6.73e+00  -1.7 7.09e-01    -  9.13e-01 3.43e-01f  1
  17  5.1213138e+01 3.16e-04 1.98e+01  -1.7 8.89e-01    -  5.25e-01 1.07e-01f  2
  18  5.0975286e+01 6.34e-05 1.67e-02  -1.7 2.38e-01    -  1.00e+00 1.00e+00f  1
  19  4.9829321e+01 3.00e-04 1.13e+00  -2.5 2.31e+00    -  4.44e-01 4.95e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  4.9289866e+01 3.35e-04 2.02e+00  -2.5 1.49e+00    -  2.45e-01 3.63e-01f  1
  21  4.8966799e+01 2.24e-04 1.71e+00  -2.5 9.28e-01    -  5.44e-01 3.48e-01f  1
  22  4.8783498e+01 2.29e-04 2.17e+00  -2.5 8.82e-01    -  1.51e-01 2.08e-01f  1
  23  4.8291840e+01 3.74e-05 2.96e+00  -2.5 4.92e-01    -  6.00e-01 1.00e+00f  1
  24  4.8282188e+01 3.63e-05 4.16e+00  -2.5 6.09e-01    -  8.09e-02 2.47e-02h  2
  25  4.8262502e+01 1.78e-05 1.39e+00  -2.5 5.14e-02    -  1.00e+00 5.52e-01h  1
  26  4.8254558e+01 1.63e-07 2.36e-04  -2.5 7.94e-03    -  1.00e+00 1.00e+00f  1
  27  4.7937142e+01 1.00e-05 5.62e-01  -3.8 4.75e-01    -  5.66e-01 6.69e-01f  1
  28  4.7820136e+01 3.34e-05 2.12e+00  -3.8 1.78e-01    -  3.89e-01 6.59e-01f  1
  29  4.7783098e+01 2.79e-05 2.90e+00  -3.8 5.92e-02    -  3.87e-01 6.26e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  4.7770434e+01 1.57e-05 1.58e+00  -3.8 2.83e-02    -  3.23e-01 4.47e-01f  1
  31  4.7749483e+01 3.30e-06 8.70e-03  -3.8 2.29e-02    -  1.00e+00 1.00e+00f  1
  32  4.7751975e+01 7.22e-08 1.99e-05  -3.8 3.37e-03    -  1.00e+00 1.00e+00h  1
  33  4.7728011e+01 5.39e-07 2.65e-02  -5.7 2.96e-02    -  8.25e-01 8.09e-01f  1
  34  4.7722475e+01 1.90e-08 2.66e-05  -5.7 5.54e-03    -  1.00e+00 1.00e+00f  1
  35  4.7722473e+01 2.57e-11 4.67e-09  -5.7 1.40e-05    -  1.00e+00 1.00e+00h  1
  36  4.7722109e+01 1.01e-10 1.02e-05  -8.6 3.65e-04    -  9.99e-01 9.99e-01f  1
  37  4.7722108e+01 1.73e-13 1.02e-12  -8.6 4.04e-07    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 37

                                   (scaled)                 (unscaled)
Objective...............:   4.7722108190710927e+01    4.7722108190710927e+01
Dual infeasibility......:   1.0231815394945443e-12    1.0231815394945443e-12
Constraint violation....:   1.7251478023894151e-13    1.7251478023894151e-13
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5059054427784873e-09    2.5059054427784873e-09
Overall NLP error.......:   2.5059054427784873e-09    2.5059054427784873e-09


Number of objective function evaluations             = 43
Number of objective gradient evaluations             = 38
Number of equality constraint evaluations            = 43
Number of inequality constraint evaluations          = 43
Number of equality constraint Jacobian evaluations   = 38
Number of inequality constraint Jacobian evaluations = 38
Number of Lagrangian Hessian evaluations             = 37
Total seconds in IPOPT                               = 1.756

EXIT: Optimal Solution Found.
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  |        0 (       0) 152.00us (  3.53us)        43
       nlp_g  | 163.00ms (  3.79ms) 117.46ms (  2.73ms)        43
  nlp_grad_f  |        0 (       0) 262.00us (  6.72us)        39
  nlp_hess_l  | 815.00ms ( 22.03ms)   1.03 s ( 27.94ms)        37
   nlp_jac_g  | 612.00ms ( 15.69ms) 497.33ms ( 12.75ms)        39
       total  |   1.76 s (  1.76 s)   1.76 s (  1.76 s)         1
```

