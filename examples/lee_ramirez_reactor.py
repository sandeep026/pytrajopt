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
    data = r"hen, Weifeng, and Lorenz T. Biegler. “Nested Direct Transcription Optimization for Singular Optimal Control Problems.” AIChE Journal, vol. 62, no. 10, Oct. 2016, pp. 3611–27. DOI.org (Crossref), https://doi.org/10.1002/aic.15272."
    return data


def lee_ramirez_reactor():
    ocp_data = {}
    ocp_data["nx"] = 7
    ocp_data["nu"] = 2
    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    ocp_data["E"] = syms.get_function("E", -xfs[0] * xfs[3], "event")
    ocp_data["e"] = syms.get_function(
        "e", cs.vertcat(x0s - cs.DM([1, 0.1, 40, 0, 0, 1, 0])), "event"
    )
    ocp_data["e_lb"] = [0] * 7
    ocp_data["e_ub"] = [0] * 7
    ocp_data["t0"] = 0
    ocp_data["tf"] = 10

    c1 = 100.0
    c2 = 0.51
    c3 = 4.0
    x1, x2, x3, x4, x5, x6, x7 = (
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xs[4],
        xs[5],
        xs[6],
    )
    u1, u2 = us[0], us[1]
    t1 = 14.35 + x3 + (x3**2) / 111.5
    t2 = 0.22 + x5
    t3 = (0.22 * x7) / t2 + x6
    g1 = (x3 * t3) / t1
    g2 = (0.233 * x3 * (0.0005 + x5)) / ((0.022 + x5) * t1)
    g3 = (0.09 * x5) / (0.034 + x5)
    dyn = cs.vertcat(
        u1 + u2,
        g1 * x2 - ((u1 + u2) * x2) / x1,
        (u1 / x1) * c1 - ((u1 + u2) * x3) / x1 - (g1 * x2) / c2,
        g2 * x2 - ((u1 + u2) * x4) / x1,
        (u2 * c3) / x1 - ((u1 + u2) * x5) / x1,
        -g3 * x6,
        g3 * (1 - x7),
    )

    ocp_data["f"] = syms.get_function("f", dyn, "path")

    inf = cs.DM.inf()
    ocp_data["x_lb"] = [1, -inf, 0, -inf, 0, -inf, -inf]
    ocp_data["x_ub"] = [cs.DM.inf()] * 7
    ocp_data["u_lb"] = [0, 0]
    ocp_data["u_ub"] = [1, 1]
    ocp_data["x_s"] = [1, 1, 50, 1, 1, 1, 1]
    ocp = Ocp(**ocp_data)

    n = 100
    method = "rk4"
    disc = disc_opt(n=n, method=method)

    nlp = cs.Opti()
    nlp.solver("ipopt")

    dis_ocp = generate_NLP(
        ocp,
        nlp,
        disc,
    )

    guess_data = {
        "n": disc.n,
        "xl": [1, 0.1, 40, 0, 0, 1, 0],
        "xr": [1, 0.1, 40, 0, 0, 1, 0],
        "ul": [0, 0],
        "ur": [1, 1],
        "tl": 0,
        "tr": 10,
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "lee_ramirez_reactor_sol")
    check_feasibility(ocp, dis_ocp, sol, "lee_ramirez_reactor_feas")


if __name__ == "__main__":
    lee_ramirez_reactor()
