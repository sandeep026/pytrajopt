import casadi as cs
import numpy as np
from trajopt import (
    Ocp,
    casadi_integrator,
    check_feasibility,
    generate_NLP,
    plot_solution,
    pc_interp
)


def solve_multiphase_ocp():
    #  Common Physics & Shared Symbols
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", 3, 1)
    us = cs.SX.sym("us", 1, 1)

    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", 3, 1)
    xfs = cs.SX.sym("xfs", 3, 1)

    #  dynamics model
    mu, ve = 1.0, 0.5
    r, v, m = xs[0, :], xs[1, :], xs[2, :]
    f = cs.vertcat(v, -mu / (r.T @ r) + us / m, -us / ve)
    f_func = cs.Function("f_shared", [ts, xs, us], [f])

    method = casadi_integrator(plugin="cvodes", nx=3, nu=1, path=[ts, xs, us], f=f)

    #  Phase Definitions
    phases_data = [
        {
            "n": 30,
            "t0": cs.DM(0),
            "tf_bounds": (1e-6, 100.0), 
            "u_bounds": (1.167, 1.167),
            "x_bounds": (
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
            ),  
            "E": cs.DM(0),
            "e": x0s - cs.DM([1.0, 0.0, 1.0]),
            "e_bounds": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "e_scale": [1.0, 1.0, 1.0],
            "t_guess": (None, 0.08),
            "X_guess": lambda n: cs.vertcat(
                cs.linspace(1.0, 1.002, n + 1).T,
                cs.linspace(0.0, 0.02, n + 1).T,
                cs.linspace(1.0, 0.8, n + 1).T,
            ),
            "U_guess": lambda n: cs.DM.ones(1, n).T * 1.167,
        },
        {
            "n": 100,
            "t0": None,
            "tf_bounds": (None, None), 
            "u_bounds": (0.0, 0.875),
            "x_bounds": ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            "E": cs.DM(0),
            "e": tfs - t0s,
            "e_bounds": ([1e-6], [100.0]), 
            "e_scale": [1.0],
            "t_guess": (0.08, 0.20),
            "X_guess": lambda n: cs.vertcat(
                cs.linspace(1.002, 1.01, n + 1).T,
                cs.linspace(0.02, 0.08, n + 1).T,
                cs.linspace(0.7, 0.52, n + 1).T,
            ),
            "U_guess": lambda n: cs.DM.ones(1, n).T * 0.5,
        },
        {
            "n": 30,
            "t0": None,
            "tf_bounds": (None, None),  
            "u_bounds": (0.0, 0.0),
            "x_bounds": ([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            "E": -xfs[0],
            "e": xfs[2] - 0.5,
            "e_bounds": ([0.0], [0.0]),
            "e_scale": [1.0],
            "t_guess": (0.20, 0.28),
            "X_guess": lambda n: cs.vertcat(
                cs.linspace(1.01, 1.011, n + 1).T,
                cs.linspace(0.08, 0.0, n + 1).T,
                cs.DM.ones(1, n + 1) * 0.5,
            ),
            "U_guess": lambda n: cs.DM.zeros(1, n).T,
        },
    ]

    # OCP Generation Loop
    ocps, dis_ocps = [], []
    nlp = cs.Opti()

    for i, data in enumerate(phases_data):
        n = data["n"]
        ocp_config = {
            "nx": 3,
            "nu": 1,
            "n": n,
            "t0": data["t0"],
            "tf": None,
            "L": None,
            "f": f_func,
            "method": method,
            "E": cs.Function(f"E{i}", [t0s, tfs, x0s, xfs], [data["E"]]),
            "e": cs.Function(f"e{i}", [t0s, tfs, x0s, xfs], [data["e"]]),
            "e_lb": cs.DM(data["e_bounds"][0]),
            "e_ub": cs.DM(data["e_bounds"][1]),
            "u_lb": cs.DM([data["u_bounds"][0]]),
            "u_ub": cs.DM([data["u_bounds"][1]]),
            "tf_lb": cs.DM([data["tf_bounds"][0]])
            if data["tf_bounds"][0] is not None
            else None,
            "tf_ub": cs.DM([data["tf_bounds"][1]])
            if data["tf_bounds"][1] is not None
            else None,
            "x_lb": cs.DM(data["x_bounds"][0]),
            "x_ub": cs.DM(data["x_bounds"][1]),
            "J_s": cs.DM([1.0]),
            "t_s": cs.DM([0.1 * (i + 1)]),
            "x_s": cs.DM([1.0, 0.1, 1.0]),
            "u_s": cs.DM([1.0]),
            "e_s": cs.DM(data["e_scale"]),
            "h_s": None,
        }

        ocp = Ocp(**ocp_config)
        dis_ocp = generate_NLP(ocp, nlp)

        dis_ocp["X_guess"] = data["X_guess"](n)
        dis_ocp["U_guess"] = data["U_guess"](n)
        dis_ocp["t_guess"] = data["t_guess"]

        ocps.append(ocp)
        dis_ocps.append(dis_ocp)

    #  Phase Linkages (Hard Knots)
    # Phase 0 -> Phase 1
    tf0, xf0 = dis_ocps[0]["tf"], dis_ocps[0]["X"][:, -1]
    t01, x01 = dis_ocps[1]["t0"], dis_ocps[1]["X"][:, 0]
    nlp.subject_to(tf0 - t01 == 0)
    nlp.subject_to(xf0[0] - x01[0] == 0)
    nlp.subject_to(xf0[1] - x01[1] == 0)
    nlp.subject_to(x01[2] == 0.7)
    nlp.subject_to(xf0[2] - x01[2] >= 0.1)
    nlp.subject_to(xf0[2] - x01[2] <= 0.3)

    # Phase 1 -> Phase 2
    tf1, xf1 = dis_ocps[1]["tf"], dis_ocps[1]["X"][:, -1]
    t02, x02 = dis_ocps[2]["t0"], dis_ocps[2]["X"][:, 0]
    nlp.subject_to(tf1 - t02 == 0)
    nlp.subject_to(xf1 - x02 == 0)

    nlp.solver("ipopt")

    for i, dis_ocp in enumerate(dis_ocps):
        t0_g, tf_g = dis_ocp["t_guess"]
        if t0_g is not None:
            nlp.set_initial(dis_ocp["t0"], t0_g)
        if tf_g is not None:
            nlp.set_initial(dis_ocp["tf"], tf_g)
        nlp.set_initial(dis_ocp["X"], dis_ocp["X_guess"])
        nlp.set_initial(dis_ocp["U"], dis_ocp["U_guess"])

    sol = nlp.solve()

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages('all.pdf') as pdf:
        for j in range(3):
            plt.figure()
            for i, dis_ocp in enumerate(dis_ocps):
                t=sol.value(dis_ocp["t"]).ravel()
                X=sol.value(dis_ocp["X"])
                if X.ndim==1:
                    plt.plot(t,X,label=f'phase {i}')
                else:
                    plt.plot(t,X[j,:].ravel(),label=f'phase {i}')
            plt.xlabel('time [s]')
            plt.ylabel(f'$x_{j}$')
            plt.grid()
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()

        for j in range(1):
            plt.figure()
            for i, dis_ocp in enumerate(dis_ocps):
                t=sol.value(dis_ocp["t"])
                U=sol.value(dis_ocp["U"])
                zoh=pc_interp(t,U)
                zoh=zoh(t)
                if zoh.ndim==1:
                    plt.plot(t,zoh,label=f'phase {i}')
                else:
                    plt.plot(t,zoh[j,:].ravel(),label=f'phase {i}')
            plt.xlabel('time [s]')
            plt.ylabel(f'$u_{j}$')
            plt.grid()
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()


#    for i, (ocp, dis_ocp) in enumerate(zip(ocps, dis_ocps), start=1):
#        plot_solution(sol, dis_ocp, name=f"s{i}")
#        check_feasibility(ocp, dis_ocp, sol, name=f"f{i}")


if __name__ == "__main__":
    solve_multiphase_ocp()
