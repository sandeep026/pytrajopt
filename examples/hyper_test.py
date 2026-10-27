import casadi as cs
import numpy as np

from pytrajopt import (
    Ocp,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
)


def get_ocp(aug: bool):
    if not aug:
        ocp_data = {}
        ocp_data["nx"] = 1
        ocp_data["nu"] = 1
        ts = cs.SX.sym("ts", 1, 1)
        xs = cs.SX.sym("xs", ocp_data["nx"], 1)
        us = cs.SX.sym("us", ocp_data["nu"], 1)
        t0s = cs.SX.sym("t0s", 1, 1)
        tfs = cs.SX.sym("tfs", 1, 1)
        x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
        xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
        path = [ts, xs, us]
        event = [t0s, tfs, x0s, xfs]
        ocp_data["L"] = cs.Function("L", path, [0.5 * (xs**2 + us**2)])
        ocp_data["E"] = None
        ocp_data["e"] = cs.Function(
            "e", event, [cs.vertcat(x0s - cs.DM([1]), xfs - 1.5)]
        )
        ocp_data["e_lb"] = cs.DM([0] * 2)
        ocp_data["e_ub"] = cs.DM([0] * 2)
        ocp_data["t0"] = cs.DM(0)
        ocp_data["tf"] = cs.DM(10000)
        ocp_data["f"] = cs.Function("f", path, [-(xs)**3 + us])
        ocp = Ocp(**ocp_data)
        path = [ts, xs, us]
        return ocp, path
    else:
        ocp_data = {}
        ocp_data["nx"] = 2
        ocp_data["nu"] = 2
        ts = cs.SX.sym("ts", 1, 1)
        xs = cs.SX.sym("xs", ocp_data["nx"], 1)
        us = cs.SX.sym("us", ocp_data["nu"], 1)
        t0s = cs.SX.sym("t0s", 1, 1)
        tfs = cs.SX.sym("tfs", 1, 1)
        x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
        xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)
        path = [ts, xs, us]
        event = [t0s, tfs, x0s, xfs]
        ocp_data["L"] = cs.Function(
            "L", path, [0.5 * (xs[0] ** 2 + us[0] ** 2) * us[1]]
        )
        ocp_data["E"] = None
        ocp_data["e"] = cs.Function(
            "e", event, [cs.vertcat(x0s - cs.DM([1, 0]), xfs - cs.DM([1.5, 10000]))]
        )
        ocp_data["e_lb"] = cs.DM([0] * 4)
        ocp_data["e_ub"] = cs.DM([0] * 4)
        ocp_data["t0"] = cs.DM(0)
        ocp_data["tf"] = cs.DM(1)
        ocp_data["f"] = cs.Function(
            "f", path, [cs.vertcat((-(xs[0])**3 + us[0]) * us[1], us[1])]
        )
        inf = cs.DM.inf()
        ocp_data["u_lb"] = cs.DM([-inf, 1e-8])
        ocp_data["u_ub"] = cs.DM([inf, inf])
        ocp = Ocp(**ocp_data)
        path = [ts, xs, us]
        return ocp, path


def set_guess(nlp: cs.Opti, dis_ocp: dict, disc: disc_opt, aug: bool = False):
    if not aug:
        tg = cs.linspace(0, 10000, disc.n + 1).T
        Xg = cs.vertcat(cs.linspace(1.5, 1, disc.n + 1).T)
        Ug = cs.vertcat(cs.linspace(1.5, 1, disc.n).T ** 3)
        nlp.set_initial(dis_ocp["t"], tg)
        nlp.set_initial(dis_ocp["X"], cs.vertcat(Xg))
        nlp.set_initial(dis_ocp["U"], cs.vertcat(Ug))
    else:
        Xg = cs.vertcat(
            cs.linspace(1, 1.5, disc.n + 1).T, cs.linspace(0, 10000, disc.n + 1).T
        )
        Ug = cs.vertcat(
            cs.linspace(1, 1.5, disc.n).T ** 3-0.5/(10000), cs.linspace(10000, 10000, disc.n).T
        )
        nlp.set_initial(dis_ocp["X"], cs.vertcat(Xg))
        nlp.set_initial(dis_ocp["U"], cs.vertcat(Ug))


def solve():
    aug = True
    ocp, path = get_ocp(aug)
    nlp = cs.Opti()
    method = 'bw_euler'
    #casadi_integrator(
    #    plugin="collocation",
    #    plugin_options={"collocation_scheme": "legendre", "interpolation_order": 1,'number_of_finite_elements':1},
    #    nx=ocp.nx,
    #    nu=ocp.nu,
    #    path=path,
    #    f=ocp.f(*path),
    #)
    n = 200
    disc = disc_opt(n=n, method=method)
    from pytrajopt import DynamicMesh

    if aug:
        mesh = None
    else:
        mesh = DynamicMesh(k1=0.001, k2=10000000)
    dis_ocp = generate_NLP(ocp, nlp, disc, mesh)
    nlp.solver("ipopt",{'ipopt.tol':1e-7})
    set_guess(nlp, dis_ocp, disc, aug)
    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "hyper_sol")
    check_feasibility(ocp, dis_ocp, sol, "hyper_feas")


if __name__ == "__main__":
    solve()
