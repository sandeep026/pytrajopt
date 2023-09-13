# Trajectory optimizer

## Optimal control problem

The solver attemps to solve the following type of optimal control problem. Point to point trajectories which are optimal and within the constraints are computed.

$$J(x,u)=\int^{t_0}_{t_f}L(x,u)\mathrm{d}t$$ $$x(0)=x_0,~x(t_f)=x_f$$ $$\dot{x}=f(x,u)$$ $$x_l\leq  x \leq x_u$$ $$u_l\leq  x \leq u_u$$

where, 
$t_0,t_f,L(x,u) \in \mathbb{R}$
$ x,x_0,xf,x_{l},x_{u}, f \in \mathbb{R}^{N_x}$
$u,u_{l},u_{u} \in \mathbb{R}^{N_u}   $

## Numerical solution

An analytical solution is not possible to derive in every case of an optimal control problem. Numerical methods are widely used to solve them. The solver uses direct methods where the continuos time optimal control problem is discretized and then optimized. This will be reffered to as transctiption of the optimal control problem.

The following discretization schemes are supported by the solver.

### Differentiation

1. Euler forward
2. Euler backward
3. Heun's method
4. Runge Kutta 45
5. Trapezoidal collocation
6. Hermite simpson collocation
7. LGL pseudospectral collocation

### Integration

1. Riemann sum
2. Trapezoidal
3. LGL quadrature

## NLP solver

## Derivatives

## Decision variable order

## Sparsity

## Custom solver

## Guess generation

## Grid convergence

## Plotter

## Analytical comparison

##