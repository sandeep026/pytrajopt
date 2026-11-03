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


def low_thrust():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    event = syms.get_event()
    path = syms.get_path()

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = None

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", event, [tfs])

    # event constraints
    e = cs.vertcat(x0s - cs.DM([1, 0, 0, 1]), xfs[0] - 4, xfs[2:, :] - cs.DM([0, 0.5]))
    ocp_data["e"] = cs.Function("e", event, [e])
    ocp_data["e_lb"] = cs.vcat([0] * 7)
    ocp_data["e_ub"] = cs.vcat([0] * 7)

    # dynamics
    f = cs.vcat(
        [
            xs[2, :],
            xs[3, :] / xs[0, :],
            xs[3, :] ** 2 / xs[0, :] - 1 / xs[0, :] ** 2 + us[0, :],
            -xs[2, :] * xs[3, :] / xs[0, :] + us[1, :],
        ]
    )
    ocp_data["f"] = cs.Function("fss", path, [f])

    disc_data = {}
    disc_data["n"] = 100
    disc_data["method"] = "rk4"
    # casadi_integrator(
    #    plugin="collocation",
    #    plugin_options={
    #        "interpolation_order": 4,
    #        "collocation_scheme": "legendre",
    #        "rootfinder":'kinsol',
    #        "number_of_finite_elements": 1,
    #        "simplify":True
    #    },
    #    nx=ocp_data["nx"],
    #    nu=ocp_data["nu"],
    #    path=[ts, xs, us],
    #    f=f,
    # )
    disc = disc_opt(**disc_data)

    ocp_data["u_lb"] = cs.DM([-0.01] * 2)
    ocp_data["u_ub"] = cs.DM([0.01] * 2)
    ocp_data["tf_lb"] = cs.DM([0])
    ocp_data["tf_ub"] = cs.DM([50])

    ocp_data["J_s"] = cs.DM([50])
    ocp_data["t_s"] = cs.DM([50])
    ocp_data["x_s"] = cs.DM([1, 10, 1e-2, 1e-1])
    ocp_data["u_s"] = cs.DM([0.01, 0.01])
    ocp_data["e_s"] = cs.DM([4, 10, 0.1, 1] + [4, 0.1, 1])
    ocp_data["h_s"] = None

    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()
    dis_ocp = generate_NLP(ocp, nlp, disc, DynamicMesh(scale=cs.DM(50)))
    nlp.solver("ipopt", {}, {"tol": 1e-5})

    guess_data = {
        "n": disc.n,
        "xl": [1, 0, 0, 1],
        "xr": [4, 0, 0, 0.5],
        "ul": [0, 0],
        "ur": [0, 0],
        "tl": 0,
        "tr": 50,
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()

    plot_solution(sol, dis_ocp, "low_thrust_sol")
    check_feasibility(ocp, dis_ocp, sol, "low_thrust_feas")


if __name__ == "__main__":
    low_thrust()
