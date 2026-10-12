import casadi as cs
from trajopt import casadi_integrator, plot_solution, TrajOpt


def low_thrust():
    ocp_data = {}
    ocp_data["nx"] = 4
    ocp_data["nu"] = 2

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
    ocp_data["tf"] = None

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", [t0s, tfs, x0s, xfs], [tfs])

    # event constraints
    e = cs.vertcat(
        x0s - cs.DM([1, 0, 0, 1]), xfs[0] - 4, xfs[2:, :] - cs.DM([0, 0.5])
    )
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0] * 7 )
    ocp_data["e_ub"] = cs.vcat([0] * 7 )

    # dynamics
    f = cs.vcat(
        [
            xs[2, :],
            xs[3, :] / xs[0, :],
            xs[3, :] ** 2 / xs[0, :] - 1 / xs[0, :] ** 2 + us[0, :],
            -xs[2, :] * xs[3, :] / xs[0, :] + us[1, :],
        ]
    )
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    print(ocp_data['f'].which_depends('i1',['o0']))

    method = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 4,
            "collocation_scheme": "legendre",
            "rootfinder":'kinsol',
            "number_of_finite_elements": 1,
            "simplify":True
        },
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )

    ocp_data["method"] = method

    ocp_data["u_lb"] = cs.DM([-0.01] * 2)
    ocp_data["u_ub"] = cs.DM([0.01] * 2)
    ocp_data['tf_lb'] = cs.DM([0])
    ocp_data['tf_ub'] = cs.DM([50])


    ocp_data["n"] = 100

    ocp_data['J_s']=cs.DM([50])
    ocp_data['t_s']=cs.DM([50])
    ocp_data['x_s']=cs.DM([4,25,0.1,1])
    ocp_data['u_s']=cs.DM([0.01,0.01])
    ocp_data['e_s']=cs.DM([4,25,0.1,1]+[4,0.1,1])
    ocp_data['h_s']=None

    trajopt = TrajOpt(**ocp_data)

    nlp = trajopt.generate_NLP()
    nlp["nlp"].solver("ipopt")

    Xg = cs.vcat(
        [
            cs.linspace(1, 4, ocp_data["n"] + 1).T,
            cs.linspace(0, 0, ocp_data["n"] + 1).T,
            cs.linspace(0, 0, ocp_data["n"] + 1).T,
            cs.linspace(1, 0.5, ocp_data["n"] + 1).T,
        ]
    )

    nlp["nlp"].set_initial(nlp["X"], Xg)
    nlp["nlp"].set_initial(nlp["U"], cs.DM.zeros(nlp["U"].shape))
    nlp["nlp"].set_initial(nlp["tf"], 46)
    sol = nlp["nlp"].solve()
    plot_solution(sol, nlp)


if __name__ == "__main__":
    low_thrust()
