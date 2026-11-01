import casadi as cs

from pytrajopt import (
    Ocp,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
)


def ally():
    ocp_data = {}
    ocp_data["nx"] = 3
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
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", event, [xfs[2]])
    ocp_data["e"] = cs.Function(
        "e",
        event,
        [cs.vertcat(x0s - cs.DM([0,1, 0]))],
    )
    ocp_data["e_lb"] = cs.DM([0] * 3)
    ocp_data["e_ub"] = cs.DM([0] * 3)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(cs.pi/2)
    ocp_data["f"] = cs.Function(
        "f",
        path,
        [cs.vertcat(xs[1],us,0.5*(xs[1]**2-xs[0]**2))],
    )
    ocp_data["x_lb"] = cs.DM([-cs.DM.inf()]*3)
    ocp_data["x_ub"] = cs.DM([cs.DM.inf()]*3)
    n = 40
    ocp_data["u_lb"] = cs.DM([-1])
    ocp_data["u_ub"] = cs.DM([1])

    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()

    method = casadi_integrator(
        plugin="rk",
        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    disc = disc_opt(n=n, method=method)
    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")
    Xg = cs.repmat(cs.DM([0,1,0]), 1, disc.n + 1)
    Ug = cs.vertcat(cs.linspace(-1, 1, n).T)
    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], Ug)
    sol = nlp.solve()
    # return {'Xg':sol.value(dis_ocp['X']),'Ug':sol.value(dis_ocp['U']),'dtg':cs.diff(ocp.t0+tau*(ocp.tf-ocp.t0))}
    plot_solution(sol, dis_ocp, "ally_chen_sol")
    check_feasibility(ocp, dis_ocp, sol, "ally_chen_feas")


ally()