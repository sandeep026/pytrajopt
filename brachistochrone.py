import casadi as cs
from trajopt import (
    Ocp,
    casadi_integrator,
    check_feasibility,
    generate_NLP,
    plot_solution,
)


def brach():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1
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
    e = cs.vertcat(x0s - cs.DM([0, 0, 0]), xfs[0:2, :] - cs.DM([1e3, 1]))

    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0] * 5)
    ocp_data["e_ub"] = cs.vcat([0] * 5)

    # dynamics
    g = 9.8
    f = cs.vcat([xs[2] * cs.sin(us), xs[2] * cs.cos(us), g * cs.cos(us)])
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    method = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 4,
            "collocation_scheme": "legendre",
            "rootfinder": "newton",
            "number_of_finite_elements": 1,
            "simplify": True,
        },
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )

    ocp_data["method"] = method

    ocp_data["u_lb"] = None
    ocp_data["u_ub"] = None
    ocp_data["tf_lb"] = None
    ocp_data["tf_ub"] = cs.DM(100)

    ocp_data["n"] = 100

    ocp_data["J_s"] = cs.DM([10])
    ocp_data["t_s"] = cs.DM([10])
    ocp_data["x_s"] = cs.DM([1e3, 160, 20])
    ocp_data["u_s"] = cs.DM([1])
    ocp_data["e_s"] = cs.DM([1e3, 160, 20] + [1e3, 160])
    ocp_data["h_s"] = None

    ocp = Ocp(**ocp_data)

    nlp = cs.Opti()

    dis_ocp = generate_NLP(ocp, nlp)
    nlp.solver("ipopt")

    Xg = cs.vcat(
        [
            cs.linspace(1, 1e3, ocp.n + 1).T,
            cs.linspace(1, 1e2, ocp.n + 1).T,
            cs.linspace(1, 3e2, ocp.n + 1).T,
        ]
    )

    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], 1e2 * cs.DM.ones(dis_ocp["U"].shape))
    nlp.set_initial(dis_ocp["tf"], 25)
    sol = nlp.solve()
    plot_solution(sol, dis_ocp)
    check_feasibility(ocp, dis_ocp, sol)


if __name__ == "__main__":
    brach()
