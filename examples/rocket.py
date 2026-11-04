import casadi as cs

from pytrajopt import (
    Ocp,
    Syms,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
    LinGuess
)

def meta():
    data = r"https://web.casadi.org/blog/nlp-scaling/"
    return data



def rocket():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    event = syms.get_event()
    path = syms.get_path()

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(100)

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", event, [x0s[-1] - xfs[-1]])

    # event constraints
    e = cs.vertcat(x0s - cs.DM([0, 0, 5e5]), xfs[0] - 1e5)

    ocp_data["e"] = cs.Function("e", event, [e])
    ocp_data["e_lb"] = cs.vcat([0] * 4)
    ocp_data["e_ub"] = cs.vcat([0] * 4)

    g = 9.81
    alp = 1 / (300 * g)

    # dynamics
    f = cs.vcat([xs[1], us / xs[2] - g, -alp * us])
    ocp_data["f"] = cs.Function("fss", path, [f])

    ocp_data["u_lb"] = cs.DM([0.0])
    ocp_data["x_lb"] = cs.DM([-cs.DM.inf(), -cs.DM.inf(), 0])

    ocp_data["J_s"] = cs.DM([3e5])
    ocp_data["x_s"] = cs.DM([1e5, 2e3, 3e5])
    ocp_data["u_s"] = cs.DM([1e8])
    ocp_data["e_s"] = cs.DM([1e5, 2e3, 3e5] + [1e5])

    ocp = Ocp(**ocp_data)

    disc_data = {}
    disc_data["n"] = 100
    disc_data["method"] = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 3,
            "collocation_scheme": "legendre",
            "rootfinder": "kinsol",
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

    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")

    guess_data = {
        "n": disc.n,
        "xl": [0, 0, 1e5],
        "xr": [0, 0, 1e5],
        "ul": [0],
        "ur": [0],
    }
    guess=LinGuess(**guess_data)
    guess.set_guess(dis_ocp,nlp)    

    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "rocket_sol")
    check_feasibility(ocp, dis_ocp, sol, "rocket_feas")


if __name__ == "__main__":
    rocket()
