import casadi as cs
import numpy as np
from trajopt import casadi_integrator, plot_solution, TrajOpt


def ffrb():
    """Free-Flying Robot OCP: Chapter 25, Example 25.1."""
    ocp_data = {}
    ocp_data["nx"] = 6
    ocp_data["nu"] = 4
    ocp_data["n"] = 1000
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(12)

    alpha = 0.2
    beta = 0.2

    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)

    t0s = cs.SX.sym("t0")
    tfs = cs.SX.sym("tf")
    x0s = cs.SX.sym("x0", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xf", ocp_data["nx"], 1)

    thrust = us[0] - us[1] + us[2] - us[3]

    f = cs.vertcat(
        xs[3],
        xs[4],
        xs[5],
        thrust * cs.cos(xs[2]),
        thrust * cs.sin(xs[2]),
        alpha * (us[0] - us[1]) - beta * (us[2] - us[3]),
    )

    ocp_data["method"] = casadi_integrator(
        plugin="rk",
        #plugin_options={
        #    "interpolation_order": 3,
        #    "collocation_scheme": "legendre",
        #    "rootfinder": "fast_newton",
        #    "number_of_finite_elements": 1,
        #    "simplify": False,
        #},
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )

    ocp_data["f"] = cs.Function(
        "f",
        [ts, xs, us],
        [f],
    )

    ocp_data["L"] = cs.Function(
        "L",
        [ts, xs, us],
        [cs.sum1(us)],
    )

    ocp_data["E"] = None

    h = cs.vertcat(
        us[0] + us[1],
        us[2] + us[3],
    )

    ocp_data["h"] = cs.Function(
        "h",
        [ts, xs, us],
        [h],
    )

    ocp_data["h_lb"] = -cs.DM.inf() * cs.DM.ones(2, 1)
    ocp_data["h_ub"] = cs.DM.ones(2, 1)

    x_initial = cs.DM(
        [
            -10,
            -10,
            np.pi / 2,
            0,
            0,
            0,
        ]
    )

    x_final = cs.DM.zeros(ocp_data["nx"], 1)

    e_expr = cs.vertcat(
        x0s - x_initial,
        xfs - x_final,
    )

    ocp_data["e"] = cs.Function(
        "e_ffrb",
        [t0s, tfs, x0s, xfs],
        [e_expr],
    )

    ocp_data["e_lb"] = cs.DM.zeros(2 * ocp_data["nx"], 1)
    ocp_data["e_ub"] = cs.DM.zeros(2 * ocp_data["nx"], 1)

    ocp_data["x_lb"] = None
    ocp_data["x_ub"] = None

    ocp_data["u_lb"] = cs.DM.zeros(ocp_data["nu"], 1)
    ocp_data["u_ub"] = cs.DM.ones(ocp_data["nu"], 1)

    trajopt = TrajOpt(**ocp_data)

    # Generate the nonlinear program
    nlp = trajopt.generate_NLP()

    Xg = cs.vcat(
        [
            cs.linspace(x_initial[i], 0, ocp_data["n"] + 1).T
            for i in range(ocp_data["nx"])
        ]
    )
    Ug = cs.vcat(
        [cs.linspace(0.1, 0.9, ocp_data["n"]).T for i in range(ocp_data["nu"])]
    )

    nlp["nlp"].set_initial(nlp["X"], Xg)
    nlp["nlp"].set_initial(nlp["U"], Ug)
    nlp["nlp"].solver(
        "ipopt",
        {"detect_simple_bounds": False},
        {
            "max_iter": 5000,
            "tol": 1e-8,
            "print_level": 5,
        },
    )

    sol = nlp["nlp"].solve()

    plot_solution(sol, nlp)


if __name__ == "__main__":
    from time import perf_counter

    t1 = perf_counter()
    ffrb()
    t2 = perf_counter()
    print("elapsed time", t2 - t1)
