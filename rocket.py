import casadi as cs
from trajopt import casadi_integrator, plot_solution, TrajOpt, check_feasibility


def rocket():
    ocp_data = {}
    ocp_data["nx"] = 3
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
    ocp_data["tf"] = cs.DM(100)

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", [t0s, tfs, x0s, xfs], [x0s[-1]-xfs[-1]])

    # event constraints
    e = cs.vertcat(
        x0s - cs.DM([0, 0, 5e5]), xfs[0] - 1e5)
    
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0]*4 )
    ocp_data["e_ub"] = cs.vcat([0]*4 )

    g=9.81
    alp=1/(300*g)

    # dynamics
    f = cs.vcat(
        [
            xs[1],
            us/xs[2]-g,
            -alp*us
        ]
    )
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])


    method = 'exp_mid'
    method = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 3,
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

    ocp_data["u_lb"] = cs.DM([0.0])
    ocp_data['x_lb'] = cs.DM([-cs.DM.inf(),-cs.DM.inf(),0])

    ocp_data["n"] = 100

    #ocp_data['J_s']=cs.DM([3e5])
    #ocp_data['x_s']=cs.DM([1e5,2e3,3e5])
    #ocp_data['u_s']=cs.DM([1e8])
    #ocp_data['e_s']=cs.DM([1e5,2e3,3e5]+[1e5])

    trajopt = TrajOpt(**ocp_data)

    nlp = trajopt.generate_NLP()
    nlp["nlp"].solver("ipopt")

    Xg = cs.vcat(
        [
            cs.linspace(0, 0*1e5, ocp_data["n"] + 1).T,
            cs.linspace(0, 0, ocp_data["n"] + 1).T,
            cs.linspace(1e5, 1e5, ocp_data["n"] + 1).T,
        ]
    )

    nlp["nlp"].set_initial(nlp["X"], Xg)
    nlp["nlp"].set_initial(nlp["U"], cs.DM.zeros(nlp["U"].shape))
    sol = nlp["nlp"].solve()
    plot_solution(sol, nlp)
    check_feasibility(ocp_data,nlp,sol)

if __name__=='__main__':
    rocket()