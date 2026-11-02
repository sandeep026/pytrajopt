import casadi as cs
import numpy as np

from pytrajopt import (
    DynamicMesh,
    Ocp,
    Syms,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
    LinGuess
)


def ffrb():
    """Free-Flying Robot OCP: Chapter 25, Example 25.1."""
    ocp_data = {}
    ocp_data["nx"] = 6
    ocp_data["nu"] = 4
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(12)

    alpha = 0.2
    beta = 0.2

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    event = syms.get_event()
    path = syms.get_path()

    thrust = us[0] - us[1] + us[2] - us[3]

    f = cs.vertcat(
        xs[3],
        xs[4],
        xs[5],
        thrust * cs.cos(xs[2]),
        thrust * cs.sin(xs[2]),
        alpha * (us[0] - us[1]) - beta * (us[2] - us[3]),
    )

    disc_data = {}
    disc_data["n"] = 50
    disc_data["method"] = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 1,
            "collocation_scheme": "legendre",
            "rootfinder": "fast_newton",
            "number_of_finite_elements": 1,
            "simplify": True,
        },
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )
    disc = disc_opt(**disc_data)

    ocp_data["f"] = cs.Function(
        "f",
        path,
        [f],
    )

    ocp_data["L"] = cs.Function(
        "L",
        path,
        [cs.sum1(us)],
    )

    ocp_data["E"] = None

    h = cs.vertcat(
        us[0] + us[1],
        us[2] + us[3],
    )

    ocp_data["h"] = cs.Function(
        "h",
        path,
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
        event,
        [e_expr],
    )

    ocp_data["e_lb"] = cs.DM.zeros(2 * ocp_data["nx"], 1)
    ocp_data["e_ub"] = cs.DM.zeros(2 * ocp_data["nx"], 1)

    ocp_data["x_lb"] = None
    ocp_data["x_ub"] = None

    ocp_data["u_lb"] = cs.DM.zeros(ocp_data["nu"], 1)
    ocp_data["u_ub"] = cs.DM.ones(ocp_data["nu"], 1)

    ocp = Ocp(**ocp_data)

    nlp = cs.Opti()

    mesh = DynamicMesh(scale=float(ocp.tf - ocp.t0), offset=0, k1=0.1, k2=2)
    # Generate the nonlinear program
    dis_ocp = generate_NLP(ocp, nlp, disc, mesh)


    guess_data = {
        "n": disc.n,
        "xl": [-10, -10, cs.pi/2, 0, 0, 0],
        "xr": [0, 0, 0, 0, 0, 0],
        "ul": [0.1]*ocp.nu,
        "ur": [0.9]*ocp.nu,
        "tl": 0,
        "tr": 12,
    }
    guess=LinGuess(**guess_data)
    guess.set_guess(dis_ocp,nlp)

    nlp.solver(
        "ipopt",
        {"ipopt.linear_solver": "mumps"},
        {
            "max_iter": 5000,
            "tol": 1e-8,
            "print_level": 5,
        },
    )

    sol = nlp.solve()

    plot_solution(sol, dis_ocp, "free_flying_robot_sol")
    check_feasibility(ocp, dis_ocp, sol, "free_flying_robot_feas")


if __name__ == "__main__":
    ffrb()
