import casadi as cs
import numpy as np

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
    data = r"Betts, John T. Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. Second, Society for Industrial and Applied Mathematics, 2010. DOI.org (Crossref), https://doi.org/10.1137/1.9780898718577."
    return data


def ffrb():
    """Free-Flying Robot OCP: Chapter 25, Example 25.1."""
    ocp_data = {}
    ocp_data["nx"] = 6
    ocp_data["nu"] = 4

    ocp_data["t0"] = 0
    ocp_data["tf"] = 12

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()

    L= cs.sum1(us)
    x_initial = cs.DM([-10,-10,cs.pi/2,0,0,0])
    x_final = cs.DM.zeros(ocp_data["nx"], 1)
    e = cs.vertcat(
        x0s - x_initial,
        xfs - x_final,
    )
    alpha = 0.2
    beta = 0.2
    thrust = us[0] - us[1] + us[2] - us[3]
    f = cs.vertcat(
        xs[3],
        xs[4],
        xs[5],
        thrust * cs.cos(xs[2]),
        thrust * cs.sin(xs[2]),
        alpha * (us[0] - us[1]) - beta * (us[2] - us[3]),
    )
    h = cs.vertcat(
        us[0] + us[1],
        us[2] + us[3],
    )

    syms.collect_expression({'L':L,'e':e,'f':f,'h':h})
    ocp_data.update(syms.generate_funcs())    
    ocp_data["e_lb"] = [0] * e.numel()
    ocp_data["e_ub"] = [0] * e.numel()
    inf=cs.DM.inf()
    ocp_data["h_lb"] = [-inf] * 2
    ocp_data["h_ub"] = [1] * 2
    ocp_data["u_lb"] = [0] * ocp_data["nu"]
    ocp_data["u_ub"] = [1] * ocp_data["nu"]
    ocp = Ocp(**ocp_data)

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

    nlp = cs.Opti()
    nlp.solver("ipopt")
    mesh = DynamicMesh(scale=12, offset=0, k1=0.1, k2=2)
    # Generate the nonlinear program
    dis_ocp = generate_NLP(ocp, nlp, disc, mesh)

    guess_data = {
        "n": disc.n,
        "xl": [-10, -10, cs.pi / 2, 0, 0, 0],
        "xr": [0, 0, 0, 0, 0, 0],
        "ul": [0.1] * ocp.nu,
        "ur": [0.9] * ocp.nu,
        "tl": 0,
        "tr": 12,
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()

    plot_solution(sol, dis_ocp, "free_flying_robot_sol")
    check_feasibility(ocp, dis_ocp, sol, "free_flying_robot_feas")


if __name__ == "__main__":
    ffrb()
