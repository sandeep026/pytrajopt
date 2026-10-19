import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from pytrajopt import Ocp, casadi_integrator, plot_solution, check_feasibility, generate_NLP, disc_opt


def cart_costate():
    # ocp_data
    ocp_data = {}

    ocp_data["nx"] = 2
    ocp_data["nu"] = 1

    # event and Mayer
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)

    # Lagrange dynamics and path
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)

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
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = zr
    ocp_data["e_ub"] = zr

    # dynamics
    f = cs.vcat([xs[1], -xs[1] + us])
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    disc_data={}
    disc_data["n"] = 50
    disc_data["method"] = "bw_euler"
    disc=disc_opt(**disc_data)

    ocp = Ocp(**ocp_data)

    nlp=cs.Opti()

    dis_ocp = generate_NLP(ocp,nlp,disc)
    nlp.solver("ipopt")
    nlp.set_initial(dis_ocp["X"], cs.DM.ones( dis_ocp["X"].shape))
    nlp.set_initial(dis_ocp["U"], cs.DM.zeros(dis_ocp["U"].shape))
    sol = nlp.solve()


    t = np.linspace(0, 2, disc_data["n"] + 1)[1:]
    out = [
        sol.value(dis_ocp['dual']["lam_x"])[0, :],
        np.ones(disc_data["n"] ),
        sol.value(dis_ocp['dual']["lam_x"])[1, :],
        1 - 0.5 * np.exp(np.linspace(0, 2, disc_data["n"] )),
    ]
    labels = [
        r"$\lambda_1$ (num)",
        r"$\lambda_1$ (ana)",
        r"$\lambda_2$ (num)",
        r"$\lambda_2$ (ana)",
    ]
    markers=['-','^','--','s']

    for i in range(4):
        plt.plot(t, out[i], markers[i],label=labels[i])
    plt.ylabel('costates (numerical vs. analytical)')
    plt.legend()
    plt.show()

    plot_solution(sol,dis_ocp,'cart_costate_sol')
    check_feasibility(ocp,dis_ocp,sol,'cart_costate_feas')

    
if __name__ == "__main__":
    cart_costate()
