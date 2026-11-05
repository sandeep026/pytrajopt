import casadi as cs

from pytrajopt import (
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
    data = r"Ross, I. M., et al. “Scaling and Balancing for High-Performance Computation of Optimal Controls.” Journal of Guidance, Control, and Dynamics, vol. 41, no. 10, Oct. 2018, pp. 2086–97. DOI.org (Crossref), https://doi.org/10.2514/1.G003382."
    return data


def brach():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1
    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()

    # initial and final time
    ocp_data["t0"] = 0
    ocp_data["tf"] = None

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = syms.get_function("E", tfs, "event")

    # event constraints
    e = cs.vertcat(x0s - cs.DM([0, 0, 0]), xfs[0:2, :] - cs.DM([1e3, 1]))

    ocp_data["e"] = syms.get_function("e", e, "event")
    ocp_data["e_lb"] = [0] * 5
    ocp_data["e_ub"] = [0] * 5

    # dynamics
    g = 9.8
    f = cs.vcat([xs[2] * cs.sin(us), xs[2] * cs.cos(us), g * cs.cos(us)])
    ocp_data["f"] = syms.get_function('f',f,'path')

    disc_data = {}

    disc_data["n"] = 20
    disc_data["method"] = "rk4"
    #cgl = -cs.hcat(
    #    [cs.cos(k * cs.pi / disc_data["n"]) for k in range(disc_data["n"] + 1)]
    #)
    # disc_data['grid']=(0.5)*(1+cgl)
    disc = disc_opt(**disc_data)

    ocp_data["tf_ub"] = 100

    ocp_data["J_s"] = 10
    ocp_data["t_s"] = 10
    ocp_data["x_s"] = [1e3, 160, 20]
    ocp_data["u_s"] = [1]
    ocp_data["e_s"] = [1e3, 160, 20] + [1e3, 160]
    ocp_data["h_s"] = None

    ocp = Ocp(**ocp_data)

    nlp = cs.Opti()

    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")

    guess_data = {
        "n": disc.n,
        "xl": [1, 1, 1],
        "xr": [1e3, 1e2, 3e2],
        "ul": [1e2],
        "ur": [1e2],
        "tr": 25,
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)
    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "brachistochrone_sol")
    check_feasibility(ocp, dis_ocp, sol, "brachistochrone_feas")


if __name__ == "__main__":
    brach()
