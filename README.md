# PyTrajOpt

**PyTrajOpt** (Python-based Trajectory Optimization) is a modern Python framework for solving complex optimal control problems (OCPs). Powered by CasADi for high-performance symbolic differentiation and Pydantic for strict runtime configuration validation, PyTrajOpt supports both single-phase and multiphase formulations via multiple shooting and direct collocation. It goes a step further by estimating state costates and performing rigorous feasibility checks to ensure the reliability of your generated solutions.


## Installation

PyTrajOpt uses [Poetry](https://python-poetry.org/) for robust dependency management and packaging. You can install the framework directly from GitHub or set it up locally for development.

### Option 1: Install Directly from GitHub via Pip

If you want to quickly install the package into your current Python environment without cloning the repository:

```bash
pip install git+https://github.com/sandeep026/pytrajopt.git

```

### Option 2: Install Directly from GitHub via Poetry

To add PyTrajOpt as a dependency in another Poetry-managed project directly from the remote repository:

```bash
poetry add git+https://github.com/sandeep026/pytrajopt.git

```

### Option 3: Local Installation for Development (Clone & Poetry)

If you want to run the provided examples or contribute to the source code:

1. **Clone the repository:**
```bash
git clone https://github.com/sandeep026/pytrajopt.git
cd pytrajopt

```

2. **Install dependencies:**
Poetry will automatically read the `pyproject.toml` configuration, resolve the dependencies (such as CasADi, Pydantic, NumPy, and Matplotlib), and set up a dedicated virtual environment:


```bash
poetry install

```

3. **Activate the virtual environment:**
```bash
poetry shell

```

## Quick Start & Usage Example

To demonstrate how to define and solve optimal control problems with PyTrajOpt, consider the classical **Low-Thrust Orbital Transfer** problem.

The problem formulation is adapted from Ross et al. (*"Low-thrust, high-accuracy trajectory optimization," Journal of Guidance, Control, and Dynamics*, 2007). The minimum time reported when solved using the Legendre-Gauss-Lobatto (LGL) pseudospectral method is **47.706** for 300 nodes.

### Problem Definition

The state and control vectors for the problem are defined as 

$$\mathbf{x}^T = \begin{bmatrix} r & \theta & v_r & v_t \end{bmatrix}$$

$$\mathbf{u}^T = \begin{bmatrix} u_r & u_t \end{bmatrix}$$

$$ \text{Minimize} ~~ J[\mathbf{x}(\cdot), \mathbf{u}(\cdot), t_f] = t_f$$

#### System Dynamics:

$$\dot{r} = v_r$$

$$\dot{\theta} = \frac{v_t}{r}$$

$$\dot{v}_r = \frac{v_t^2}{r} - \frac{1}{r^2} + u_r$$

$$\dot{v}_t = -\frac{v_r v_t}{r} + u_t$$

#### Boundary Conditions & Constraints:

$$(r, \theta, v_r, v_t)(0) = (1, 0, 0, 1)$$

$$(r, v_r, v_t)(t_f) = (4, 0, 0.5)$$

$$-0.01 \le \mathbf{u} \le 0.01$$

*(Primal and dual solutions for this problem can be found in the accompanying [low_thrust_sol.pdf](./examples/low_thrust_sol.pdf) and*
*results of feasibility checks are provided in [low_thrust_feas.pdf](./examples/low_thrust_feas.pdf)*


### Usage Example (`examples/low_thrust.py`)

Here is how you can set up, solve, and visualize this problem using PyTrajOpt's Pydantic-backed configuration and CasADi transcription engine:

```python
import casadi as cs
from pytrajopt import casadi_integrator, plot_solution, Ocp, check_feasibility, generate_NLP


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
        x0s - cs.DM([1, 0, 0, 1]), xfs[0] - 4, xfs[2:, :] - cs.DM([0, 0.5])
    )
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0] * 7 )
    ocp_data["e_ub"] = cs.vcat([0] * 7 )

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
    ocp_data['tf_lb'] = cs.DM([0])
    ocp_data['tf_ub'] = cs.DM([50])


    ocp_data["n"] = 100

    ocp_data['J_s']=cs.DM([50])
    ocp_data['t_s']=cs.DM([50])
    ocp_data['x_s']=cs.DM([1,10,1e-2,1e-1])
    ocp_data['u_s']=cs.DM([0.01,0.01])
    ocp_data['e_s']=cs.DM([4,10,0.1,1]+[4,0.1,1])
    ocp_data['h_s']=None

    ocp = Ocp(**ocp_data)
    nlp=cs.Opti()
    dis_ocp = generate_NLP(ocp,nlp)
    nlp.solver("ipopt")
    Xg = cs.vcat(
        [
            cs.linspace(1, 4,   ocp.n + 1).T,
            cs.linspace(0, 0,   ocp.n + 1).T,
            cs.linspace(0, 0,   ocp.n + 1).T,
            cs.linspace(1, 0.5, ocp.n + 1).T,
        ]
    )
    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], cs.DM.zeros(dis_ocp["U"].shape))
    nlp.set_initial(dis_ocp["tf"], 46)
    sol = nlp.solve()

    plot_solution(sol, dis_ocp,'low_thrust_sol') 
    check_feasibility(ocp,dis_ocp,sol,'low_thrust_feas')
    

if __name__ == "__main__":
    low_thrust()

```

> **Explore More Examples:** For complete, ready-to-run scripts covering single-phase problems (like the Brachistochrone and rocket launches) as well as complex multiphase staging problems, check out the `examples/` folder in the repository.
> 
>

Output

```shell
******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.2.

Number of nonzeros in equality constraint Jacobian...:     2907
Number of nonzeros in inequality constraint Jacobian.:      605
Number of nonzeros in Lagrangian Hessian.............:     2595

Total number of variables............................:      605
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        0
                     variables with only upper bounds:        0
Total number of equality constraints.................:      407
Total number of inequality constraints...............:      605
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:      201
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  9.2000000e-01 5.25e+00 1.83e-03  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  9.1999901e-03 2.14e+00 7.81e+01  -1.0 2.88e+01    -  5.48e-02 6.21e-01h  1
   2  1.0023819e-02 2.14e+00 9.08e+03  -1.0 8.66e+02    -  1.09e-03 8.27e-04h  1
   3  1.0531395e-02 2.14e+00 1.39e+04  -1.0 1.06e+03    -  1.30e-03 4.90e-04h  1
   4  1.0708156e-02 2.14e+00 2.03e+04  -1.0 4.13e+03    -  1.91e-04 1.60e-04h  1
   5  1.0768356e-02 2.14e+00 2.60e+04  -1.0 1.15e+04    -  1.17e-04 5.49e-05h  1
   6  1.0794242e-02 2.14e+00 3.36e+04  -1.0 3.23e+04    -  2.94e-05 2.38e-05h  1
   7r 1.0794242e-02 2.14e+00 9.99e+02   0.3 0.00e+00    -  0.00e+00 2.96e-07R  5
   8r 4.2969919e-02 1.38e+00 9.98e+02   0.3 4.60e+03    -  5.16e-04 1.37e-03f  1
   9  4.2991715e-02 1.38e+00 3.01e+01  -1.0 2.38e+03    -  4.53e-04 1.51e-05h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  4.3078514e-02 1.38e+00 5.21e+02  -1.0 1.04e+04    -  7.78e-05 5.06e-05h  1
  11r 4.3078514e-02 1.38e+00 9.99e+02   0.1 0.00e+00    -  0.00e+00 3.79e-07R  5
  12r 9.0446322e-02 6.82e-01 9.97e+02   0.1 2.27e+03    -  2.15e-03 2.24e-03f  1
  13  9.0477082e-02 6.82e-01 6.26e-01  -1.0 1.64e+03    -  9.83e-06 1.60e-05h  1
  14  9.3363170e-02 6.81e-01 1.25e+03  -1.0 1.08e+03    -  1.18e-06 1.83e-03f  1
  15  9.6744598e-02 6.79e-01 1.23e+03  -1.0 6.08e+02    -  4.48e-04 2.21e-03f  1
  16  1.0300510e-01 6.76e-01 1.09e+03  -1.0 3.65e+02    -  6.46e-04 4.10e-03f  1
  17  1.0979368e-01 6.72e-01 9.67e+02  -1.0 2.85e+02    -  1.07e-03 4.40e-03f  1
  18  1.1651106e-01 6.68e-01 8.56e+02  -1.0 1.81e+02    -  2.56e-03 4.19e-03f  1
  19  1.2617612e-01 6.63e-01 6.86e+02  -1.0 1.07e+02    -  3.27e-03 5.79e-03f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.4750800e-01 6.56e-01 1.31e+02  -1.0 8.92e+01    -  4.56e-03 1.36e-02f  1
  21  1.5947749e-01 6.52e-01 9.97e+01  -1.0 8.89e+01    -  8.99e-04 8.17e-03h  1
  22  1.6717085e-01 6.48e-01 1.03e+02  -1.0 6.16e+01    -  1.39e-02 5.60e-03f  1
  23  1.7948200e-01 6.44e-01 1.12e+02  -1.0 6.33e+01    -  1.26e-04 9.16e-03h  1
  24  1.9734135e-01 6.38e-01 1.33e+02  -1.0 5.38e+01   0.0 8.78e-03 1.22e-02f  1
  25  2.0212085e-01 6.38e-01 1.33e+02  -1.0 8.29e+02  -0.5 2.65e-04 1.10e-03f  1
  26  2.0229741e-01 6.38e-01 1.33e+02  -1.0 9.98e+01  -0.1 3.21e-03 9.17e-05h  1
  27  2.1020231e-01 6.37e-01 1.50e+02  -1.0 1.21e+03  -0.5 2.30e-04 1.42e-03f  1
  28  2.1051485e-01 6.37e-01 1.50e+02  -1.0 1.06e+02  -0.1 2.15e-03 1.63e-04h  1
  29  2.1146862e-01 6.37e-01 1.51e+02  -1.0 2.64e+04  -0.6 1.98e-06 8.00e-06f  2
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  2.3179092e-01 6.33e-01 1.98e+02  -1.0 1.80e+02  -0.2 5.05e-03 9.57e-03f  1
  31  2.3242283e-01 6.33e-01 1.98e+02  -1.0 4.95e+01   0.3 1.59e-03 3.47e-04h  1
  32  2.4165975e-01 6.32e-01 2.05e+02  -1.0 3.20e+02  -0.2 1.04e-04 2.85e-03h  1
  33  2.4326663e-01 6.31e-01 2.05e+02  -1.0 5.61e+01   0.2 6.35e-03 8.69e-04h  1
  34  2.4720131e-01 6.31e-01 2.06e+02  -1.0 8.36e+02  -0.3 6.92e-04 5.36e-04h  1
  35  2.5260904e-01 6.29e-01 2.05e+02  -1.0 5.97e+01   0.2 2.01e-03 2.72e-03f  1
  36  2.6235195e-01 6.26e-01 2.00e+02  -1.0 1.56e+01   0.6 3.84e-03 5.79e-03f  1
  37  2.6868276e-01 6.24e-01 1.98e+02  -1.0 5.66e+01   0.1 3.91e-03 3.18e-03f  1
  38  2.8044110e-01 6.19e-01 1.95e+02  -1.0 1.33e+01   0.5 3.72e-03 7.15e-03f  1
  39  2.8254165e-01 6.19e-01 1.94e+02  -1.0 5.34e+01   0.1 6.61e-03 1.07e-03f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  40  2.9650621e-01 6.13e-01 1.92e+02  -1.0 1.75e+01   0.5 9.09e-03 8.75e-03f  1
  41  3.0773229e-01 6.10e-01 1.91e+02  -1.0 8.55e+01   0.0 6.03e-03 5.96e-03f  1
  42  3.2463350e-01 6.03e-01 1.91e+02  -1.0 1.73e+01   0.4 1.35e-02 1.07e-02f  1
  43  3.3529502e-01 6.01e-01 1.92e+02  -1.0 1.22e+02  -0.0 2.30e-03 3.99e-03f  1
  44  3.5976559e-01 5.92e-01 1.92e+02  -1.0 2.05e+01   0.4 1.62e-02 1.53e-02f  1
  45  3.6043165e-01 5.91e-01 1.92e+02  -1.0 2.21e+02  -0.1 3.25e-03 2.00e-04h  1
  46  3.9267828e-01 5.80e-01 1.91e+02  -1.0 3.06e+01   0.3 1.31e-02 2.07e-02f  1
  47  4.0276953e-01 5.76e-01 1.91e+02  -1.0 9.78e+00   0.8 1.29e-02 6.97e-03h  1
  48  4.2360672e-01 5.69e-01 1.89e+02  -1.0 2.89e+01   0.3 5.52e-03 1.28e-02f  1
  49  4.3531039e-01 5.65e-01 1.88e+02  -1.0 9.46e+00   0.7 1.26e-02 8.30e-03h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  50  4.4131063e-01 5.63e-01 1.87e+02  -1.0 2.89e+01   0.2 1.42e-02 3.62e-03h  1
  51  4.5841280e-01 5.56e-01 1.87e+02  -1.0 1.39e+01   0.7 1.48e-02 1.25e-02f  1
  52  4.8421922e-01 5.49e-01 1.84e+02  -1.0 3.73e+01   0.2 1.73e-03 1.56e-02f  1
  53  5.0535806e-01 5.41e-01 1.83e+02  -1.0 1.06e+01   0.6 9.29e-03 1.59e-02f  1
  54  5.2362382e-01 5.37e-01 1.80e+02  -1.0 3.58e+01   0.1 1.56e-03 9.55e-03f  1
  55  5.4218535e-01 5.30e-01 1.77e+02  -1.0 1.04e+01   0.6 1.83e-02 1.46e-02h  1
  56  5.5379926e-01 5.27e-01 1.74e+02  -1.0 4.15e+01   0.1 1.26e-03 6.01e-03f  1
  57  5.7864073e-01 5.17e-01 1.69e+02  -1.0 1.13e+01   0.5 1.58e-02 2.07e-02f  1
  58  5.9652439e-01 5.11e-01 1.63e+02  -1.0 3.82e+01   0.0 1.47e-03 1.11e-02f  1
  59  5.9841709e-01 5.11e-01 1.63e+02  -1.0 1.19e+01   0.5 1.46e-02 1.73e-03h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  60  6.1519966e-01 5.07e-01 1.60e+02  -1.0 6.22e+01  -0.0 1.29e-03 8.58e-03f  1
  61  6.7326747e-01 4.84e-01 1.55e+02  -1.0 1.23e+01   0.4 4.07e-03 5.07e-02f  1
  62  6.9479983e-01 4.80e-01 1.53e+02  -1.0 7.61e+01  -0.1 3.90e-03 8.53e-03h  1
  63  7.1075656e-01 4.74e-01 1.51e+02  -1.0 1.21e+01   0.4 3.21e-02 1.41e-02f  1
  64  7.2859889e-01 4.70e-01 1.49e+02  -1.0 3.39e+01  -0.1 1.85e-03 7.86e-03h  1
  65  8.1918119e-01 4.34e-01 1.20e+02  -1.0 8.48e+00   0.3 2.24e-02 8.24e-02f  1
  66  8.4437698e-01 4.28e-01 1.16e+02  -1.0 2.39e+01  -0.2 1.74e-02 1.47e-02h  1
  67  9.2714653e-01 3.95e-01 8.89e+01  -1.0 7.22e+00   0.3 6.34e-02 9.11e-02f  1
  68  9.6754052e-01 3.83e-01 7.98e+01  -1.0 1.27e+01  -0.2 2.04e-02 3.99e-02f  1
  69  9.9968127e-01 3.68e-01 7.55e+01  -1.0 4.94e+00   0.2 1.80e-02 4.66e-02f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  70  1.0000027e+00 3.67e-01 1.21e+02  -1.0 9.71e+00  -0.3 6.24e-02 2.79e-03h  1
  71  9.9996106e-01 3.47e-01 1.85e+02  -1.0 5.09e+00   0.2 8.24e-02 6.76e-02f  1
  72  9.9996100e-01 3.42e-01 1.14e+02  -1.0 1.16e+01  -0.3 6.19e-02 9.22e-02f  1
  73  9.9996648e-01 3.30e-01 3.62e+02  -1.0 4.21e+00   0.1 1.09e-01 4.04e-02h  1
  74  9.9996546e-01 3.24e-01 3.49e+02  -1.0 1.20e+01  -0.4 2.87e-02 3.09e-02h  1
  75  9.9996631e-01 3.18e-01 8.60e+02  -1.0 4.20e+00   0.1 2.04e-01 1.92e-02h  1
  76  9.9996437e-01 3.08e-01 7.91e+02  -1.0 1.07e+01  -0.4 8.71e-02 8.14e-02f  1
  77  9.9996546e-01 2.89e-01 1.10e+03  -1.0 3.99e+00  -0.0 2.94e-01 7.23e-02h  1
  78  9.9996318e-01 2.70e-01 9.75e+02  -1.0 6.92e+00  -0.5 1.25e-01 1.06e-01f  1
  79  9.9996247e-01 2.70e-01 9.39e+02  -1.0 4.66e+01  -1.0 2.64e-02 1.52e-02h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  80  9.9995980e-01 2.43e-01 7.30e+02  -1.0 4.65e+00  -0.5 2.97e-01 1.17e-01h  1
  81  9.9995708e-01 2.25e-01 6.08e+02  -1.0 8.15e+00  -1.0 1.45e-01 7.61e-02h  1
  82  9.9995652e-01 2.23e-01 6.17e+02  -1.0 1.94e+02  -1.5 8.31e-04 9.60e-03h  1
  83  9.9995112e-01 1.95e-01 4.36e+02  -1.0 4.43e+00  -2.0 2.20e-01 1.24e-01f  1
  84  9.9995113e-01 1.93e-01 4.31e+02  -1.0 1.62e+02    -  2.22e-03 6.00e-03h  1
  85  9.9994502e-01 1.54e-01 3.69e+02  -1.0 2.48e+00  -0.6 1.26e-01 2.05e-01h  1
  86  9.9993947e-01 1.34e-01 1.49e+02  -1.0 5.72e+00  -1.1 3.67e-01 1.30e-01f  1
  87  9.9990737e-01 6.06e-02 1.46e+02  -1.0 4.45e+00  -1.6 4.27e-01 5.47e-01h  1
  88  9.9988251e-01 4.05e-02 1.01e+02  -1.0 7.06e+00  -2.1 3.19e-01 3.23e-01f  1
  89  9.9985996e-01 3.16e-02 7.56e+01  -1.0 5.88e+00  -2.5 2.42e-01 2.39e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  90  9.9983960e-01 2.58e-02 5.62e+01  -1.0 1.05e+01  -3.0 2.23e-01 2.12e-01f  1
  91  9.9984049e-01 2.52e-02 5.26e+01  -1.0 1.13e+02  -3.5 5.91e-03 1.58e-02f  2
  92  9.9979170e-01 1.49e-02 7.34e+01  -1.0 2.71e+00  -2.2 3.28e-01 5.95e-01h  1
  93  9.9977234e-01 1.51e-02 6.25e+01  -1.0 4.22e+00    -  2.22e-01 2.90e-01f  1
  94  9.9975477e-01 1.18e-02 4.42e+01  -1.0 3.04e+00    -  3.26e-01 3.94e-01f  1
  95  9.9973427e-01 2.22e-03 1.36e+01  -1.0 1.30e+00    -  6.79e-01 9.86e-01h  1
  96  9.9973861e-01 3.51e-05 6.35e-03  -1.0 7.54e-01    -  1.00e+00 1.00e+00f  1
  97  9.9973800e-01 1.91e-07 8.91e-02  -2.5 1.18e-02    -  1.00e+00 1.00e+00h  1
  98  9.9971519e-01 1.48e-04 1.60e-04  -2.5 3.75e-01    -  1.00e+00 1.00e+00f  1
  99  9.9968820e-01 9.49e-08 4.15e-01  -3.8 3.69e-03    -  9.60e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 100  9.7471349e-01 1.05e-02 1.94e-02  -3.8 1.04e+00    -  7.22e-01 8.49e-01f  1
 101  9.7837233e-01 2.24e-03 3.61e-02  -3.8 1.07e+00    -  1.00e+00 7.96e-01h  1
 102  9.8107389e-01 1.53e-03 1.70e-04  -3.8 9.91e-01    -  1.00e+00 1.00e+00f  1
 103  9.8163061e-01 3.74e-04 3.71e-04  -3.8 4.30e-01    -  1.00e+00 1.00e+00h  1
 104  9.7066033e-01 6.00e-03 2.72e-03  -5.7 2.50e+00    -  3.01e-01 4.41e-01h  1
 105  9.6644363e-01 4.49e-03 2.05e-03  -5.7 1.84e+00    -  2.55e-01 2.93e-01h  1
 106  9.6240545e-01 1.40e-03 6.56e-04  -5.7 1.32e-01  -2.6 7.49e-01 6.83e-01h  1
 107  9.5849454e-01 6.06e-04 2.22e-04  -5.7 3.13e-01  -3.1 7.44e-01 5.92e-01h  1
 108  9.5671398e-01 5.31e-04 2.06e-03  -5.7 5.16e-01  -3.6 1.00e+00 4.73e-01h  1
 109  9.5607251e-01 4.36e-04 2.13e-03  -5.7 9.84e-01  -4.1 8.50e-01 4.46e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 110  9.5590288e-01 3.98e-04 1.12e-03  -5.7 1.82e+00  -4.6 7.80e-01 1.31e-01h  1
 111  9.5551537e-01 8.35e-04 9.87e-04  -5.7 2.01e+00  -5.0 7.45e-01 3.14e-01f  1
 112  9.5529262e-01 2.22e-03 1.84e-03  -5.7 4.91e+00    -  1.07e-01 3.01e-01f  1
 113  9.5493466e-01 2.48e-03 6.87e-03  -5.7 4.47e+00    -  4.02e-01 2.36e-01f  1
 114  9.5489775e-01 2.07e-03 5.83e-03  -5.7 1.19e+00    -  1.00e+00 1.66e-01h  1
 115  9.5481998e-01 1.28e-04 5.82e-04  -5.7 7.40e-02    -  1.00e+00 9.43e-01h  1
 116  9.5480891e-01 1.53e-06 5.94e-06  -5.7 2.81e-02    -  1.00e+00 1.00e+00f  1
 117  9.5448782e-01 2.28e-05 1.66e-05  -8.6 1.60e-01    -  8.93e-01 8.77e-01h  1
 118  9.5444346e-01 2.66e-07 2.52e-07  -8.6 1.18e-02    -  1.00e+00 1.00e+00h  1
 119  9.5444346e-01 4.46e-10 1.52e-11  -8.6 1.43e-04    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 119

                                   (scaled)                 (unscaled)
Objective...............:   9.5444345906140005e-01    9.5444345906140005e-01
Dual infeasibility......:   1.5186740753847516e-11    1.5186740753847516e-11
Constraint violation....:   4.4618753136660416e-10    4.4618753136660416e-10
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   2.5059339191123623e-09    2.5059339191123623e-09
Overall NLP error.......:   2.5059339191123623e-09    2.5059339191123623e-09


Number of objective function evaluations             = 134
Number of objective gradient evaluations             = 120
Number of equality constraint evaluations            = 134
Number of inequality constraint evaluations          = 134
Number of equality constraint Jacobian evaluations   = 122
Number of inequality constraint Jacobian evaluations = 122
Number of Lagrangian Hessian evaluations             = 119
Total seconds in IPOPT                               = 5.307

EXIT: Optimal Solution Found.
      solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
       nlp_f  |        0 (       0) 551.00us (  4.11us)       134
       nlp_g  | 412.00ms (  3.07ms) 324.21ms (  2.42ms)       134
  nlp_grad_f  |   1.00ms (  8.26us)   1.34ms ( 11.05us)       121
  nlp_hess_l  |   2.80 s ( 23.91ms)   3.11 s ( 26.59ms)       117
   nlp_jac_g  |   1.69 s ( 13.76ms)   1.45 s ( 11.81ms)       123
       total  |   5.32 s (  5.32 s)   5.32 s (  5.32 s)         1
```


## Repository Structure

The project is structured into a clean modular layout separating the core package, comprehensive examples, test suite, and configuration files:

```text
pytrajopt/
├── pytrajopt/              # Core package source code
│   ├── pytrajopt.py        # Pydantic configuration & CasADi NLP transcription engine
│   └── pub_qual.mplstyle   # Publication-quality Matplotlib style sheet
├── examples/               # Example optimal control problems & generated PDF solutions
│   ├── brachistochrone.py  # Brachistochrone curve problem
│   ├── cart_costate.py     # Cart costate estimation example
│   ├── container_crane.py  # Container crane maneuver
│   ├── free_flying_robot.py# Free-flying space robot trajectory
│   ├── low_thrust.py       # Low-thrust orbital transfer
│   ├── multiphase_rocket.py# Multiphase rocket staging problem
│   ├── rocket.py           # Single-phase rocket launch
│   └── shuttle_cross_range.py # Space shuttle cross-range maneuver
├── tests/                  # Automated testing suite
│   └── test_trajopt.py     # Unit and integration tests
├── pyproject.toml          # Poetry dependency and project configuration
├── poetry.lock             # Locked dependency versions
├── LICENSE                 # Open-source license agreement
└── README.md               # Project documentation

```

## Mathematical Formulation & Transcription

### Multiphase OCP

A multiphase optimal control problem consists of $N_{\text{phases}}$ sequential or interconnected phases, indexed by $p \in \{1, \dots, N_{\text{phases}}\}$. The framework solves problems of the general form:

$$\min_{\{x^{(p)}(\cdot), u^{(p)}(\cdot), t_0^{(p)}, t_f^{(p)}\}_{p=1}^{N_{\text{phases}}}} \quad J = \sum_{p=1}^{N_{\text{phases}}} \left( \int_{t_0^{(p)}}^{t_f^{(p)}} L^{(p)}(t, x^{(p)}, u^{(p)})\mathrm{d}t + E^{(p)}(t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_f^{(p)}) \right)$$

Subject to the following dynamics and constraints for each phase $p = 1, \dots, N_{\text{phases}}$:

$$\dot{x}^{(p)}(t) = f^{(p)}(t, x^{(p)}, u^{(p)}), \quad t \in [t_0^{(p)}, t_f^{(p)}]$$

$$e_{lb}^{(p)} \leq e^{(p)}(t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_f^{(p)}) \leq e_{ub}^{(p)}$$

$$h_{lb}^{(p)} \leq h^{(p)}(t, x^{(p)}, u^{(p)}) \leq h_{ub}^{(p)}$$

$$x_{lb}^{(p)} \leq x^{(p)}(t) \leq x_{ub}^{(p)}, \quad u_{lb}^{(p)} \leq u^{(p)}(t) \leq u_{ub}^{(p)}$$

And across-phase linkage constraints (e.g., state continuity, time coordination, or event conditions):

$$g_{lb} \leq g\left(\{t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_f^{(p)}\}_{p=1}^{N_{\text{phases}}}\right) \leq g_{ub}$$

### Variable & Function Definitions 

| Name | Description | Shape |
| --- | --- | --- |
| $N_{\text{phases}}$ | Total number of phases | $\mathbb{Z}^+$ |
| $p$ | Phase index ($p \in \{1, \dots, N_{\text{phases}}\}$) | $\mathbb{Z}^+$ |
| $t$ | Independent variable (Time) within phase $p$ | $\mathbb{R}$ |
| $t_0^{(p)}$ | Initial time of phase $p$ | $\mathbb{R}$ |
| $t_f^{(p)}$ | Final time of phase $p$ | $\mathbb{R}$ |
| $x^{(p)}(t)$ | State vector in phase $p$ | $\mathbb{R}^{n_x^{(p)}}$ |
| $x_0^{(p)}$ | Initial state of phase $p$ | $\mathbb{R}^{n_x^{(p)}}$ |
| $x_f^{(p)}$ | Final state of phase $p$ | $\mathbb{R}^{n_x^{(p)}}$ |
| $u^{(p)}(t)$ | Control input vector in phase $p$ | $\mathbb{R}^{n_u^{(p)}}$ |
| $J$ | Total objective function / Cost functional | $\mathbb{R}$ |
| $L^{(p)}$ | Running cost (Lagrange term) for phase $p$ | $\mathbb{R}$ |
| $E^{(p)}$ | Terminal cost (Mayer term) for phase $p$ | $\mathbb{R}$ |
| $f^{(p)}$ | System dynamics function for phase $p$ | $\mathbb{R}^{n_x^{(p)}}$ |
| $e^{(p)}$ | Boundary / Endpoint constraint function for phase $p$ | $\mathbb{R}^{n_e^{(p)}}$ |
| $h^{(p)}$ | Path constraint function for phase $p$ | $\mathbb{R}^{n_h^{(p)}}$ |
| $g$ | Phase linkage / transition constraint function | $\mathbb{R}^{n_g}$ |
| $x_{lb}^{(p)}, x_{ub}^{(p)}$ | State lower and upper bounds for phase $p$ | $\mathbb{R}^{n_x^{(p)}}$ |
| $u_{lb}^{(p)}, u_{ub}^{(p)}$ | Control lower and upper bounds for phase $p$ | $\mathbb{R}^{n_u^{(p)}}$ |
| $e_{lb}^{(p)}, e_{ub}^{(p)}$ | Boundary constraint bounds for phase $p$ | $\mathbb{R}^{n_e^{(p)}}$ |
| $h_{lb}^{(p)}, h_{ub}^{(p)}$ | Path constraint bounds for phase $p$ | $\mathbb{R}^{n_h^{(p)}}$ |
| $g_{lb}, g_{ub}$ | Linkage constraint bounds | $\mathbb{R}^{n_g}$ |

### Transcription: From Multiphase OCP to NLP

To solve the infinite-dimensional continuous multiphase OCP on a digital computer, the problem is **transcribed** into a finite-dimensional **Nonlinear Program (NLP)** by applying direct collocation or multiple shooting independently across each phase while enforcing phase linkage constraints.

#### A. Time Grid Discretization

For each phase $p \in \{1, \dots, N_{\text{phases}}\}$, the continuous time horizon $[t_0^{(p)}, t_f^{(p)}]$ is divided into $n^{(p)}$ intervals of length $\Delta t^{(p)} = \frac{t_f^{(p)} - t_0^{(p)}}{n^{(p)}}$, generating $n^{(p)} + 1$ mesh grid points:

$$t_k^{(p)} = t_0^{(p)} + k \Delta t^{(p)}, \quad k = 0, 1, \dots, n^{(p)}$$

*(Note: For sequential phases, the final time of phase $p$ typically coincides with the initial time of phase $p+1$, i.e., $t_f^{(p)} = t_0^{(p+1)}$.)*

#### B. NLP Decision Vector ($W$) Structure

The continuous functions $x^{(p)}(t)$ and $u^{(p)}(t)$ for every phase are parameterized by state values at the grid nodes and control values over intervals. The global NLP decision variable vector $W$ is assembled by concatenating the free time variables, state matrices, and control matrices across all phases:

$$W = \begin{bmatrix} T_{\text{free}}^T & \text{vec}(X^{(1)})^T & \text{vec}(U^{(1)})^T & \dots & \text{vec}(X^{(N_{\text{phases}})})^T & \text{vec}(U^{(N_{\text{phases}})})^T \end{bmatrix}^T$$

* **Time variables ($T_{\text{free}}$):** Includes all free initial, final, or duration times across the phases (e.g., $t_0^{(1)}, t_f^{(1)}, t_f^{(2)}, \dots, t_f^{(N_{\text{phases}})}$). Fixed times are omitted from $W$.
* **State Matrix ($X^{(p)}$) for phase $p$:** Dimensions $n_x^{(p)} \times (n^{(p)} + 1)$, flattened into a column vector containing all states across all nodes in phase $p$:

$$X^{(p)} = \begin{bmatrix} x_0^{(p)} & x_1^{(p)} & \dots & x_{n^{(p)}}^{(p)} \end{bmatrix}$$

* **Control Matrix ($U^{(p)}$) for phase $p$:** Dimensions $n_u^{(p)} \times n^{(p)}$, flattened into a column vector containing control inputs across all intervals in phase $p$:

$$U^{(p)} = \begin{bmatrix} u_0^{(p)} & u_1^{(p)} & \dots & u_{n^{(p)}-1}^{(p)} \end{bmatrix}$$

#### C. Objective Function Discretization

The continuous cost functional is discretized by summing the approximated integrals (Lagrange terms) and terminal costs (Mayer terms) across all phases:

$$J_{\text{NLP}}(W) = \sum_{p=1}^{N_{\text{phases}}} \left( \frac{t_f^{(p)} - t_0^{(p)}}{n^{(p)}} \sum_{k=0}^{n^{(p)}-1} L^{(p)}(t_k^{(p)}, x_k^{(p)}, u_k^{(p)}) + E^{(p)}(t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_{n^{(p)}}^{(p)}) \right)$$

*(Note: Depending on the chosen numerical quadrature rule—such as trapezoidal or Simpson's rule—appropriate quadrature weights replace the uniform interval scaling factor).*

#### D. Constraint Vector ($G(W)$) Structure

All constraints across all phases are stacked vertically into a single global vector $G(W)$ bounded by lower and upper limits ($G_{lb} \leq G(W) \leq G_{ub}$):

1. **Phase Boundary / Endpoint Constraints:**
Evaluated at the start and end of each phase $p$:

$$G_{\text{event}}^{(p)} = e^{(p)}(t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_{n^{(p)}}^{(p)}), \quad e_{lb}^{(p)} \leq G_{\text{event}}^{(p)} \leq e_{ub}^{(p)}$$

2. **Inter-Phase Linkage Constraints:**
Enforces boundary, transition, or matching conditions between connected phases (e.g., state continuity $x_f^{(p)} = x_0^{(p+1)}$ or time synchronization):

$$G_{\text{link}} = g\left(\{t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_{n^{(p)}}^{(p)}\}_{p=1}^{N_{\text{phases}}}\right), \quad g_{lb} \leq G_{\text{link}} \leq g_{ub}$$

3. **System Dynamics / Defect Constraints:**
Enforces physical consistency between nodes using the chosen integration method independently for each phase $p$ (e.g., $x_{k+1}^{(p)} - x_k^{(p)} - \Delta t^{(p)} f^{(p)}(\cdot) = 0$):

$$G_{\text{dyn}}^{(p)} = \text{Defects}^{(p)}(X^{(p)}, U^{(p)}, t^{(p)}) = 0, \quad \forall p \in \{1, \dots, N_{\text{phases}}\}$$

4. **Path Constraints:**
Evaluated across all grid nodes $k = 0, \dots, n^{(p)}$ for each phase $p$ (using zero-order hold extension for control inputs at the final node):

$$G_{\text{path}}^{(p)} = \begin{bmatrix} h^{(p)}(t_0^{(p)}, x_0^{(p)}, u_0^{(p)}) \\ \vdots \\ h^{(p)}(t_{n^{(p)}}^{(p)}, x_{n^{(p)}}^{(p)}, u_{n^{(p)}-1}^{(p)}) \end{bmatrix}, \quad h_{lb}^{(p)} \leq G_{\text{path}}^{(p)} \leq h_{ub}^{(p)}$$

####  Assembled Final Multiphase NLP Formulation

Combining the discretized objective function, stacked multiphase constraints, and decision vector bounds, the fully assembled **Nonlinear Program (NLP)** solved by CasADi and IPOPT takes the following standard mathematical form:

$$\min_{W} \quad J_{\text{NLP}}(W)$$

Subject to the composite constraints:

$$G_{lb} \leq G(W) \leq G_{ub}$$

where the global constraint vector $G(W)$ aggregates the boundary, linkage, dynamics (defects), and path constraints across all phases:

$$G(W) = \begin{bmatrix} e^{(1)}(t_0^{(1)}, t_f^{(1)}, x_0^{(1)}, x_{n^{(1)}}^{(1)}) \\  \vdots \\  e^{(N_{\text{phases}})}(t_0^{(N_{\text{phases}})}, t_f^{(N_{\text{phases}})}, x_0^{(N_{\text{phases}})}, x_{n^{(N_{\text{phases}})}}^{(N_{\text{phases}})}) \\  g(\{t_0^{(p)}, t_f^{(p)}, x_0^{(p)}, x_{n^{(p)}}^{(p)}\}_{p=1}^{N_{\text{phases}}}) \\ \text{Defects}^{(1)}(X^{(1)}, U^{(1)}, t^{(1)}) \\  \vdots \\  \text{Defects}^{(N_{\text{phases}})}(X^{(N_{\text{phases}})}, U^{(N_{\text{phases}})}, t^{(N_{\text{phases}})}) \\  h^{(1)}(t_0^{(1)}, x_0^{(1)}, u_0^{(1)}) \\  \vdots \\  h^{(N_{\text{phases}})}(t_{n^{(N_{\text{phases}})}}^{(N_{\text{phases}})}, x_{n^{(N_{\text{phases}})}}^{(N_{\text{phases}})}, u_{n^{(N_{\text{phases}})}-1}^{(N_{\text{phases}})}) \end{bmatrix}$$

With corresponding lower and upper constraint bounds:

$$G_{lb} = \begin{bmatrix} e_{lb}^{(1)} \\ \vdots \\ e_{lb}^{(N_{\text{phases}})} \\ g_{lb} \\ \mathbf{0}^{(1)} \\ \vdots \\ \mathbf{0}^{(N_{\text{phases}})} \\ h_{lb}^{(1)} \\ \vdots \\ h_{lb}^{(N_{\text{phases}})} \end{bmatrix}, \quad G_{ub} = \begin{bmatrix} e_{ub}^{(1)} \\ \vdots \\ e_{ub}^{(N_{\text{phases}})} \\ g_{ub} \\ \mathbf{0}^{(1)} \\ \vdots \\ \mathbf{0}^{(N_{\text{phases}})} \\ h_{ub}^{(1)} \\ \vdots \\ h_{ub}^{(N_{\text{phases}})} \end{bmatrix}$$

And subject to simple variable bounds on the global decision vector $W$:

$$W_{lb} \leq W \leq W_{ub}$$

where:

$$W = \begin{bmatrix} T_{\text{free}} \\  \text{vec}(X^{(1)}) \\  \text{vec}(U^{(1)}) \\  \vdots \\  \text{vec}(X^{(N_{\text{phases}})}) \\  \text{vec}(U^{(N_{\text{phases}})}) \end{bmatrix}, \quad W_{lb} = \begin{bmatrix} T_{\text{free}, lb} \\  \text{repmat}(x_{lb}^{(1)}, 1, n^{(1)}+1) \\  \text{repmat}(u_{lb}^{(1)}, 1, n^{(1)}) \\  \vdots \\  \text{repmat}(x_{lb}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}+1) \\  \text{repmat}(u_{lb}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}) \end{bmatrix}, \quad W_{ub} = \begin{bmatrix} T_{\text{free}, ub} \\  \text{repmat}(x_{ub}^{(1)}, 1, n^{(1)}+1) \\  \text{repmat}(u_{ub}^{(1)}, 1, n^{(1)}) \\  \vdots \\  \text{repmat}(x_{ub}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}+1) \\  \text{repmat}(u_{ub}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}) \end{bmatrix}$$

## Discretization & Integration Methods

The framework supports flexible transcription strategies where **each phase can independently configure** its own integration and collocation backend, categorized into **Direct Multiple Shooting** and **Direct Collocation**:

* **Direct Multiple Shooting (Explicit & Adaptive Integrators):**
* `fw_euler`: Forward Euler (explicit 1st-order).
* `exp_mid`: Explicit Midpoint (2nd-order Runge-Kutta).
* `rk4`: Standard 4th-order Runge-Kutta (via CasADi integrator).
* `cvodes`: Adaptive-step ODE/DAE integrator from SUNDIALS (ideal for stiff dynamics).
* `collocation`: Implicit collocation polynomials via CasADi.


* **Direct Collocation (Implicit Methods):**
* `bw_euler`: Backward Euler (implicit 1st-order, highly stable).
* `trap`: Trapezoidal collocation (implicit 2nd-order).
* `her_sim`: Hermite-Simpson collocation (cubic polynomial approximation).


### Post-Processing & Solution Verification

Once the nonlinear program is successfully solved, PyTrajOpt provides robust built-in post-processing tools for visualization, costate analysis, and trajectory validation:

* **Visualization (`plot_solution` & `plot_solution_all`):**
* `plot_solution`: Generates a comprehensive PDF containing high-quality plots of the optimal states, control inputs, estimated costates, and Hamiltonian for a given phase.
* `plot_solution_all`: Designed specifically for multiphase problems, this function seamlessly aggregates and plots all sequential phases within a single, unified figure for complete trajectory inspection.

* **Costate Estimation:**
* For **Direct Multiple Shooting**, state costate estimates are automatically extracted from the KKT multipliers of the solved NLP.
* *Note:* Costate extraction for direct collocation methods is currently unimplemented and will yield empty plots for those specific schemes.

* **Feasibility Check (`check_feasibility`):**
* To rigorously verify solution accuracy, the `check_feasibility` function integrates the continuous-time system dynamics using the initial conditions and zero-order hold (ZOH) control inputs.
* It then plots these continuous trajectories alongside the discrete states obtained directly from the NLP solver.
* **Validation Tip:** Large deviations between the integrated continuous dynamics and the discrete NLP states indicate that the mesh is insufficiently resolved and requires further refinement.



