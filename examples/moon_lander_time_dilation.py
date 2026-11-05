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
    data=r'Meditch, J.. (1964). On the problem of optimal thrust programming for a lunar soft landing. IEEE Transactions on Automatic Control, 9(4), 477–484. doi:10.1109/tac.1964.1105758 https://gpops2.com/examples/moonlander.html'
    return data

def moon():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()

    ocp_data["E"] = syms.get_function('E',xfs[2],'event')
    ocp_data["e"] = syms.get_function('e',cs.vertcat(x0s - cs.DM([10, -2, 0, 0]), xfs[0:2, :] - cs.DM([0] * 2)),'event')
    ocp_data["e_lb"] = [0] * 6
    ocp_data["e_ub"] = [0] * 6
    ocp_data["t0"] = 0
    ocp_data["tf"] = 1
    ocp_data["f"] = syms.get_function('f',cs.vertcat(us[1] * xs[1], us[1] * (us[0] - 1.5), us[1] * (us[0]), us[1]),'path')
    inf=cs.DM.inf()
    ocp_data["x_lb"] = [0,   -inf, -inf, 0]
    ocp_data["x_ub"] = [inf, 0,    inf,  100]
    ocp_data["u_lb"] = [0, 0.825]
    ocp_data["u_ub"] = [3, 8.26]

    ocp = Ocp(**ocp_data)
    
    n = 40
    method = casadi_integrator(
        plugin="rk",
        plugin_options={"number_of_finite_elements": 1, "simplify": True},
        nx=ocp.nx,
        nu=ocp.nu,
        path=[ts, xs, us],
        f=ocp.f(ts, xs, us),
    )
    disc = disc_opt(n=n, method=method)
    
    nlp = cs.Opti()
    nlp.solver("ipopt")

    dis_ocp = generate_NLP(ocp, nlp, disc)

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
