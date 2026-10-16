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

$$
W = \begin{bmatrix} T_{\text{free}} \\  \text{vec}(X^{(1)}) \\  \text{vec}(U^{(1)}) \\  \vdots \\  \text{vec}(X^{(N_{\text{phases}})}) \\  \text{vec}(U^{(N_{\text{phases}})}) \end{bmatrix}, \quad W_{lb} = \begin{bmatrix} T_{\text{free}, lb} \\  \text{repmat}(x_{lb}^{(1)}, 1, n^{(1)}+1) \\  \text{repmat}(u_{lb}^{(1)}, 1, n^{(1)}) \\  \vdots \\  \text{repmat}(x_{lb}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}+1) \\  \text{repmat}(u_{lb}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}) \end{bmatrix}, \quad W_{ub} = \begin{bmatrix} T_{\text{free}, ub} \\  \text{repmat}(x_{ub}^{(1)}, 1, n^{(1)}+1) \\  \text{repmat}(u_{ub}^{(1)}, 1, n^{(1)}) \\  \vdots \\  \text{repmat}(x_{ub}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}+1) \\  \text{repmat}(u_{ub}^{(N_{\text{phases}})}, 1, n^{(N_{\text{phases}})}) \end{bmatrix}
$$

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
