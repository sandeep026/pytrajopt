# CasADi TrajOpt

A lightweight, robust, and strongly typed **Trajectory Optimization (TrajOpt)** framework built in Python using **CasADi** for symbolic automatic differentiation and **Pydantic** for runtime configuration validation.

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

## 2. Discretization & Integration Methods

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

## 3. Repository Structure

```text
├── trajopt.py   # Core Pydantic configuration & CasADi NLP transcription engine
└── example.py   # Example script demonstrating setup, solving, and plotting

```

---

## 4. Installation

Ensure you have Python installed along with the required dependencies:

```bash
pip install casadi pydantic numpy matplotlib

```

---

## 5. Quick Start & Usage Example

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
