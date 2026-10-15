import casadi as cs
import numpy as np
from trajopt import Ocp, generate_NLP, casadi_integrator, check_feasibility, plot_solution


def crane():
    # ensure initial condition placed first in event constraint to estimate correct costate at t0
    # ocp_data
    ocp_data = {}

    ocp_data["nx"] = 6
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

    c1 = 2.83374
    c2 = -0.80865
    c3 = 0.71265
    c4 = 17.2656
    c5 = 27.0756
    rho = 0.01

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(9)
    ocp_data["t0_lb"] = None
    ocp_data["tf_lb"] = None
    ocp_data["t0_ub"] = None
    ocp_data["tf_ub"] = None

    # lagrange and mayer objective
    ocp_data["L"] = cs.Function(
        "L",
        [ts, xs, us],
        [0.5 * (xs[2] ** 2 + xs[5] ** 2 + rho * (us[0] ** 2 + us[1] ** 2))],
    )
    ocp_data["E"] = None

    # event constraints
    e = cs.vertcat(
        x0s - cs.DM([0, 22, 0, 0, -1, 0]), xfs - cs.DM([10, 14, 0, 2.5, 0, 0])
    )
    zr = cs.DM.zeros(e.shape)
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = zr
    ocp_data["e_ub"] = zr

    # dynamics
    f = cs.vcat(
        [
            xs[3],
            xs[4],
            xs[5],
            us[0] + c4 * xs[2],
            us[1],
            -(us[0] + c5 * xs[2] + 2 * xs[4] * xs[5]) / xs[1],
        ]
    )
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    # path constraint
    ocp_data["h"] = None
    ocp_data["h_lb"] = None
    ocp_data["h_ub"] = None

    # simple bounds
    inf = cs.DM.inf()
    ocp_data["x_lb"] = cs.DM([-inf] * 3 + [-2.5, -1] + [-inf])
    ocp_data["x_ub"] = cs.DM([+inf] * 3 + [2.5, 1] + [inf])
    ocp_data["u_lb"] = cs.DM([-c1, c2])
    ocp_data["u_ub"] = cs.DM([c1, c3])

    ocp_data["n"] = 100
    ocp_data["method"] = "her_sim"

    ocp = Ocp(**ocp_data)
    nlp=cs.Opti()
    dis_ocp = generate_NLP(ocp,nlp)
    nlp.solver("ipopt")
    nlp.set_initial(dis_ocp["X"], cs.DM.ones( dis_ocp["X"].shape))
    nlp.set_initial(dis_ocp["U"], cs.DM.zeros(dis_ocp["U"].shape))
    sol = nlp.solve()
    plot_solution(sol, dis_ocp)
    check_feasibility(ocp, dis_ocp, sol)


if __name__ == "__main__":
    crane()
