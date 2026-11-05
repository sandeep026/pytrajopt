import casadi as cs

from pytrajopt import (
    DynamicMesh,
    LinGuess,
    Ocp,
    Syms,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
)

def meta():
    data=r'Ross, I. Michael, et al. “Low-Thrust, High-Accuracy Trajectory Optimization.” Journal of Guidance, Control, and Dynamics, vol. 30, no. 4, July 2007, pp. 921–33. DOI.org (Crossref), https://doi.org/10.2514/1.23181.'
    return data

def low_thrust():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()

    # initial and final time
    ocp_data["t0"] = 0
    ocp_data["tf"] = None

    # mayer objective
    ocp_data["E"] = syms.get_function('E',tfs,'event')

    # event constraints
    e = cs.vertcat(x0s - cs.DM([1, 0, 0, 1]), xfs[0] - 4, xfs[2:, :] - cs.DM([0, 0.5]))
    ocp_data["e"] = syms.get_function('e',e,'event')
    ocp_data["e_lb"] = [0] * 7
    ocp_data["e_ub"] = [0] * 7

    # dynamics
    f = cs.vcat(
        [
            xs[2, :],
            xs[3, :] / xs[0, :],
            xs[3, :] ** 2 / xs[0, :] - 1 / xs[0, :] ** 2 + us[0, :],
            -xs[2, :] * xs[3, :] / xs[0, :] + us[1, :],
        ]
    )
    ocp_data["f"] = syms.get_function('f',f,'path')

    disc_data = {}
    disc_data["n"] = 100
    disc_data["method"] = "rk4"
    disc = disc_opt(**disc_data)

    ocp_data["u_lb"] = [-0.01] * 2
    ocp_data["u_ub"] = [0.01] * 2
    ocp_data["tf_lb"] = [0]
    ocp_data["tf_ub"] = [50]

    ocp_data["J_s"] = [50]
    ocp_data["t_s"] = [50]
    ocp_data["x_s"] = [1, 10, 1e-2, 1e-1]
    ocp_data["u_s"] = [0.01, 0.01]
    ocp_data["e_s"] = [4, 10, 0.1, 1] + [4, 0.1, 1]
    ocp_data["h_s"] = None

    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
    dis_ocp = generate_NLP(ocp, nlp, disc, DynamicMesh(scale=50))
    nlp.solver("ipopt", {}, {"tol": 1e-5})

    guess_data = {
        "n": disc.n,
        "xl": [1, 0, 0, 1],
        "xr": [4, 0, 0, 0.5],
        "ul": [0, 0],
        "ur": [0, 0],
        "tl": 0,
        "tr": 50,
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()

    plot_solution(sol, dis_ocp, "low_thrust_sol")
    check_feasibility(ocp, dis_ocp, sol, "low_thrust_feas")


if __name__ == "__main__":
    low_thrust()
