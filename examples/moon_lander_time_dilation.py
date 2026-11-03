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


def moon():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    event = syms.get_event()
    path = syms.get_path()

    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", event, [xfs[2]])
    ocp_data["e"] = cs.Function(
        "e",
        event,
        [cs.vertcat(x0s - cs.DM([10, -2, 0, 0]), xfs[0:2, :] - cs.DM([0] * 2))],
    )
    ocp_data["e_lb"] = cs.DM([0] * 6)
    ocp_data["e_ub"] = cs.DM([0] * 6)
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(1)
    ocp_data["f"] = cs.Function(
        "f",
        path,
        [cs.vertcat(us[1] * xs[1], us[1] * (us[0] - 1.5), us[1] * (us[0]), us[1])],
    )
    ocp_data["x_lb"] = cs.DM([0, -cs.DM.inf(), -cs.DM.inf(), 0])
    ocp_data["x_ub"] = cs.DM([cs.DM.inf(), 0, cs.DM.inf(), 100])
    n = 40
    ocp_data["u_lb"] = cs.DM([0, 0.825])
    ocp_data["u_ub"] = cs.DM([3, 8.26])

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

    guess_data = {
        "n": disc.n,
        "xl": [10, -2, 0, 0],
        "xr": [10, -2, 0, 0],
        "ul": [0, 0.1],
        "ur": [3, 1],
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "moon_sol")
    check_feasibility(ocp, dis_ocp, sol, "moon_feas")


if __name__ == "__main__":
    moon()
