import casadi as cs

from pytrajopt import (
    DynamicMesh,
    Ocp,
    Syms,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
    LinGuess,
)


def lee_ramirez_reactor():
    ocp_data = {}
    ocp_data["nx"] = 7
    ocp_data["nu"] = 2
    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    event = syms.get_event()
    path = syms.get_path()
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", event, [-xfs[0] * xfs[3]])
    ocp_data["e"] = cs.Function(
        "e",
        event,
        [cs.vertcat(x0s - cs.DM([1, 0.1, 40, 0, 0, 1, 0]))],
    )
    ocp_data["e_lb"] = cs.DM([0] * 7)
    ocp_data["e_ub"] = cs.DM([0] * 7)
    ocp_data["h"] = cs.Function("h", path, [cs.vertcat(xs[0], xs[3])])
    inf = cs.DM.inf()
    ocp_data["h_lb"] = cs.DM([0] * 2)
    ocp_data["h_ub"] = cs.DM([inf] * 2)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(10)

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

    ocp_data["f"] = cs.Function(
        "f",
        path,
        [dyn],
    )
    ocp_data["x_lb"] = cs.DM([-cs.DM.inf()] * 7)
    ocp_data["x_ub"] = cs.DM([cs.DM.inf()] * 7)
    n = 30
    ocp_data["u_lb"] = cs.DM([0, 0])
    ocp_data["u_ub"] = cs.DM([1, 1])
    # ocp_data['x_s']= cs.DM([1,0.1,40,1,1,1,1])

    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()

    method = "rk4"
    # casadi_integrator(
    #    plugin="rk",
    #    plugin_options={"number_of_finite_elements": 1, "simplify": True},
    #    nx=ocp.nx,
    #    nu=ocp.nu,
    #    path=[ts, xs, us],
    #    f=ocp.f(ts, xs, us),
    # )
    disc = disc_opt(n=n, method=method)

    dis_ocp = generate_NLP(ocp, nlp, disc,DynamicMesh(scale=10) )
    nlp.solver("ipopt")

    guess_data = {
        "n": disc.n,
        "xl": [1, 0.1, 40, 0, 0, 1, 0],
        "xr": [1, 0.1, 40, 0, 0, 1, 0],
        "ul": [0, 1],
        "ur": [1, 1],
        "tl": 0,
        "tr": 10,
    }
    guess=LinGuess(**guess_data)
    guess.set_guess(dis_ocp,nlp)

    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "lee_ramirez_reactor_sol")
    check_feasibility(ocp, dis_ocp, sol, "lee_ramirez_reactor_feas")


lee_ramirez_reactor()
