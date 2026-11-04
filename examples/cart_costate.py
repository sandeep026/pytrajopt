import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

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
    data=r'Conway, B. A., and K. M. Larson. “Collocation Versus Differential Inclusion in Direct Optimization.” Journal of Guidance, Control, and Dynamics, vol. 21, no. 5, Sept. 1998, pp. 780–85. DOI.org (Crossref), https://doi.org/10.2514/2.4306.'
    return data

def cart_costate():
    # ocp_data
    ocp_data = {}

    ocp_data["nx"] = 2
    ocp_data["nu"] = 1

    syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()
    event = syms.get_event()
    path = syms.get_path()

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(2)

    # lagrange and mayer objective
    ocp_data["L"] = cs.Function(
        "L",
        [ts, xs, us],
        [us**2],
    )
    ocp_data["E"] = None

    # event constraints
    e = cs.vertcat(
        x0s - cs.DM([0, 0]),
        xfs[0]
        - 0.5 * (cs.exp(1) ** 2 - 2) * xfs[1]
        - (-cs.exp(4) + 8 * cs.exp(2) - 23) / 16,
    )
    zr = cs.DM.zeros(e.shape)
    ocp_data["e"] = cs.Function("e", event, [e])
    ocp_data["e_lb"] = zr
    ocp_data["e_ub"] = zr

    # dynamics
    f = cs.vcat([xs[1], -xs[1] + us])
    ocp_data["f"] = cs.Function("fss", path, [f])

    disc_data = {}
    disc_data["n"] = 200
    disc_data["method"] = "fw_euler"
    disc = disc_opt(**disc_data)

    ocp = Ocp(**ocp_data)

    nlp = cs.Opti()

    dis_ocp = generate_NLP(
        ocp,
        nlp,
        disc,
    )
    nlp.solver("ipopt")

    guess_data = {
        "n": disc.n,
        "xl": [1, 1],
        "xr": [1, 1],
        "ul": [0],
        "ur": [0],
        "tl": 0,
        "tr": 2,
    }
    guess = LinGuess(**guess_data)
    guess.set_guess(dis_ocp, nlp)

    sol = nlp.solve()

    t = np.linspace(0, 2, disc_data["n"] + 1)[1:]
    out = [
        sol.value(dis_ocp.dual["lam_f"])[0, :],
        np.ones(disc_data["n"]),
        sol.value(dis_ocp.dual["lam_f"])[1, :],
        1 - 0.5 * np.exp(np.linspace(0, 2, disc_data["n"])),
    ]
    labels = [
        r"$\lambda_1$ (num)",
        r"$\lambda_1$ (ana)",
        r"$\lambda_2$ (num)",
        r"$\lambda_2$ (ana)",
    ]
    markers = ["-", "^", "--", "s"]

    for i in range(4):
        plt.plot(t, out[i], markers[i], label=labels[i])
    plt.ylabel("costates (numerical vs. analytical)")
    plt.legend()
    plt.show()

    plot_solution(sol, dis_ocp, "cart_costate_sol")
    check_feasibility(ocp, dis_ocp, sol, "cart_costate_feas")


if __name__ == "__main__":
    cart_costate()
