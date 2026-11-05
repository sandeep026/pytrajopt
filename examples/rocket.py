import casadi as cs

from pytrajopt import (
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
    data = r"https://web.casadi.org/blog/nlp-scaling/"
    return data


def rocket():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()

    # initial and final time
    ocp_data["t0"] = 0
    ocp_data["tf"] = 100

    # lagrange and mayer objective
    ocp_data["E"] = syms.get_function('E',x0s[-1] - xfs[-1],'event')

    # event constraints
    e = cs.vertcat(x0s - cs.DM([0, 0, 5e5]), xfs[0] - 1e5)

    ocp_data["e"] = syms.get_function('e',e,'event')
    ocp_data["e_lb"] = [0] * 4
    ocp_data["e_ub"] = [0] * 4

    g = 9.81
    alp = 1 / (300 * g)

    # dynamics
    f = cs.vcat([xs[1], us / xs[2] - g, -alp * us])
    ocp_data["f"] = syms.get_function('f',f,'path')

    ocp_data["u_lb"] = [0.0]
    inf=cs.DM.inf()
    ocp_data["x_lb"] = [-inf, -inf, 0]

    ocp_data["J_s"] = [3e5]
    ocp_data["x_s"] = [1e5, 2e3, 3e5]
    ocp_data["u_s"] = [1e8]
    ocp_data["e_s"] = [1e5, 2e3, 3e5] + [1e5]

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
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "rocket_sol")
    check_feasibility(ocp, dis_ocp, sol, "rocket_feas")


if __name__ == "__main__":
    rocket()
