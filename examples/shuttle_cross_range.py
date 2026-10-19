import casadi as cs
from pytrajopt import casadi_integrator, plot_solution, Ocp, check_feasibility,generate_NLP, disc_opt

def shuttle_max_crossrange():
    ocp_data = {}
    ocp_data["nx"] = 5  
    ocp_data["nu"] = 2  

    # Event and Mayer variables
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)

    # Lagrange dynamics and path
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)

    # Initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = None  

    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", [t0s, tfs, x0s, xfs], [-xfs[1]])

    # Event constraints (Initial & Final conditions)
    deg2rad = cs.pi / 180.0
    e = cs.vertcat(
        x0s - cs.DM([260000.0, 0.0, 25600.0, -1.0 * deg2rad, 90.0 * deg2rad]),
        xfs[0] - 80000.0,
        xfs[2] - 2500.0,
        xfs[3] - (-5.0 * deg2rad)
    )
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0.0] * 8)
    ocp_data["e_ub"] = cs.vcat([0.0] * 8)

    Re = 20902900.0
    mu = 0.14076539e17
    rho0 = 0.002378
    hr = 23800.0
    S = 2690.0
    w = 203000.0
    g0 = 32.174
    mass = w / g0

    a0, a1 = -0.20704, 0.029244
    b0, b1, b2 = 0.07854, -0.61592e-2, 0.621408e-3
    c0, c1, c2, c3 = 1.0672181, -0.19213774e-1, 0.21286289e-3, -0.10117249e-5

    h = xs[0]
    theta = xs[1]
    v = xs[2]
    gamma = xs[3]
    psi = xs[4]
    alpha = us[0]
    beta = us[1]

    r = Re + h
    g = mu / (r**2)
    rho_a = rho0 * cs.exp(-h / hr)
    
    alpha_deg = (180.0 / cs.pi) * alpha
    cL = a0 + a1 * alpha_deg
    cD = b0 + b1 * alpha_deg + b2 * (alpha_deg**2)
    q_heat=c0 + c1 * alpha_deg + c2 * (alpha_deg**2)+c3*(alpha_deg**3)
    
    D = 0.5 * cD * rho_a * (v**2) * S
    L = 0.5 * cL * rho_a * (v**2) * S

    # Dynamics equations
    dh = v * cs.sin(gamma)
    d_theta = (v / r) * cs.cos(gamma) * cs.cos(psi)
    dv = -D / mass - g * cs.sin(gamma)
    d_gamma = (L / (mass * v)) * cs.cos(beta) + cs.cos(gamma) * (v / r - g / v)
    d_psi = (1.0 / (mass * v * cs.cos(gamma))) * (L * cs.sin(beta)) + (v / r) * cs.cos(gamma) * cs.sin(psi) * cs.tan(theta)

    f = cs.vertcat(dh, d_theta, dv, d_gamma, d_psi)
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    disc_data={}
    disc_data["n"] = 100
    disc_data['method'] = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 4,
            "collocation_scheme": "radau",
            "rootfinder": 'kinsol',
            "number_of_finite_elements": 1,
            "simplify": True
        },
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )
    disc=disc_opt(**disc_data)

    #state bounds
    ocp_data['x_lb']=cs.DM([0,-89*deg2rad,1,-89*deg2rad,-cs.DM.inf()])
    ocp_data['x_ub']=cs.DM([cs.DM.inf(),89*deg2rad,cs.DM.inf(),89*deg2rad,cs.DM.inf()])

    # Control bounds
    ocp_data["u_lb"] = cs.DM([-90.0 * deg2rad, -90.0 * deg2rad])
    ocp_data["u_ub"] = cs.DM([90.0 * deg2rad, 1.0 * deg2rad])
    
    # Final time bounds (tF <= 2500)
    ocp_data['tf_lb'] = cs.DM([0.0])
    ocp_data['tf_ub'] = cs.DM([2500.0])


    # Nominal scaling values
    ocp_data['J_s'] = cs.DM([1.0])
    ocp_data['t_s'] = cs.DM([2000.0])
    ocp_data['x_s'] = cs.DM([260000.0, 1.0, 25600.0, 1.0, 1.0])
    ocp_data['u_s'] = cs.DM([1.0, 1.0])
    ocp_data['e_s'] = cs.DM([260000.0, 1.0, 25600.0, 1.0, 1.0, 80000.0, 2500.0, 1.0])
    ocp_data['h_s'] = None  

    ocp = Ocp(**ocp_data)

    nlp=cs.Opti()

    dis_ocp = generate_NLP(ocp,nlp,disc)
    nlp.solver("ipopt")
    Xg = cs.vcat(
        [
            cs.linspace(260000.0, 80000.0,              disc.n + 1).T,
            cs.linspace(0.0, 0.5,                       disc.n + 1).T,
            cs.linspace(25600.0, 2500.0,                disc.n + 1).T,
            cs.linspace(-1.0 * deg2rad, -5.0 * deg2rad, disc.n + 1).T,
            cs.linspace(90.0 * deg2rad, 0.0,            disc.n + 1).T,
        ]
    )

    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], cs.DM.zeros(dis_ocp["U"].shape))
    nlp.set_initial(dis_ocp["tf"], 2000.0)
    
    sol = nlp.solve()
    plot_solution(sol, dis_ocp,'shuttle_cross_range_sol')
    check_feasibility(ocp, dis_ocp, sol,'shuttle_cross_range_feas')

if __name__ == "__main__":
    shuttle_max_crossrange()