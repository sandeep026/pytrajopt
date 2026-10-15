import casadi as cs
import numpy as np
from trajopt import casadi_integrator, plot_solution, Ocp, check_feasibility, generate_NLP

def solve_multiphase_ocp():
    # ----------------------------------------------------
    # 1. Common Dynamics & Physics Definition
    # ----------------------------------------------------
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", 3, 1)
    us = cs.SX.sym("us", 1, 1)

    mu = 1.0
    ve = 0.5
    r = xs[0, :]
    v = xs[1, :]
    m = xs[2, :]
    T = us
    rd = v
    rsq = r.T @ r
    vd = -mu / rsq + T / m
    md = -T / ve
    f = cs.vertcat(rd, vd, md)
    f_func = cs.Function("f_shared", [ts, xs, us], [f])

    # Collocation method with interpolation_order = 3 and n = 30 intervals per phase
    n_intervals = 50
    method = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 3,
            "collocation_scheme": "radau",
            "rootfinder": 'kinsol',
            "number_of_finite_elements": 20,
            "simplify": True
        },
        nx=3,
        nu=1,
        path=[ts, xs, us],
        f=f,
    )

    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", 3, 1)
    xfs = cs.SX.sym("xfs", 3, 1)

    # ----------------------------------------------------
    # 2. Phase Definitions (Matching Ross & Fahroo, 2004)[cite: 8]
    # ----------------------------------------------------
    
    # --- Phase 0: Stage 1 Burn[cite: 8] ---
    ocp_data0 = {}
    ocp_data0["nx"] = 3
    ocp_data0["nu"] = 1
    ocp_data0["t0"] = cs.DM(0.0)
    ocp_data0["tf"] = None
    ocp_data0["L"] = None
    ocp_data0["E"] = cs.Function("E0", [t0s, tfs, x0s, xfs], [cs.DM(0)])

    e0 = x0s - cs.DM([1.0, 0.0, 1.0])  # r(0)=1.0, v(0)=0, m(0)=1.0[cite: 8]
    ocp_data0["e"] = cs.Function("e0", [t0s, tfs, x0s, xfs], [e0])
    ocp_data0["e_lb"] = cs.DM([0.0, 0.0, 0.0])
    ocp_data0["e_ub"] = cs.DM([0.0, 0.0, 0.0])

    ocp_data0["f"] = f_func
    ocp_data0["method"] = method
    ocp_data0["u_lb"] = cs.DM([1.167])  # T_max^1 = 1.167[cite: 8]
    ocp_data0["u_ub"] = cs.DM([1.167])
    ocp_data0["tf_lb"] = cs.DM([0.01])
    ocp_data0["tf_ub"] = cs.DM([0.15])
    ocp_data0["n"] = n_intervals

    ocp_data0['J_s'] = cs.DM([1.0])
    ocp_data0['t_s'] = cs.DM([0.1])
    ocp_data0['x_s'] = cs.DM([1.0, 0.1, 1.0])
    ocp_data0['u_s'] = cs.DM([1.0])
    ocp_data0['e_s'] = cs.DM([1.0, 1.0, 1.0])
    ocp_data0['h_s'] = None
    ocp0 = Ocp(**ocp_data0)

    # --- Phase 1: Stage 2 Burn[cite: 8] ---
    ocp_data1 = {}
    ocp_data1["nx"] = 3
    ocp_data1["nu"] = 1
    ocp_data1["t0"] = None
    ocp_data1["tf"] = None
    ocp_data1["L"] = None
    ocp_data1["E"] = cs.Function("E1", [t0s, tfs, x0s, xfs], [cs.DM(0)])

    e1 = tfs - t0s
    ocp_data1["e"] = cs.Function("e1", [t0s, tfs, x0s, xfs], [e1])
    ocp_data1["e_lb"] = cs.DM([1e-6])
    ocp_data1["e_ub"] = cs.DM([0.3])

    ocp_data1["f"] = f_func
    ocp_data1["method"] = method
    ocp_data1["u_lb"] = cs.DM([0.0])
    ocp_data1["u_ub"] = cs.DM([0.875])  # T_max^2 = 0.875[cite: 8]
    ocp_data1["tf_lb"] = cs.DM([0.05])
    ocp_data1["tf_ub"] = cs.DM([0.25])
    ocp_data1["n"] = n_intervals

    ocp_data1['J_s'] = cs.DM([1.0])
    ocp_data1['t_s'] = cs.DM([0.2])
    ocp_data1['x_s'] = cs.DM([1.0, 0.1, 1.0])
    ocp_data1['u_s'] = cs.DM([1.0])
    ocp_data1['e_s'] = cs.DM([1.0])
    ocp_data1['h_s'] = None
    ocp1 = Ocp(**ocp_data1)

    # --- Phase 2: Coast Arc (Zero Thrust) & Max Radius[cite: 8] ---
    ocp_data2 = {}
    ocp_data2["nx"] = 3
    ocp_data2["nu"] = 1
    ocp_data2["t0"] = None
    ocp_data2["tf"] = None
    ocp_data2["L"] = None
    ocp_data2["E"] = cs.Function("E2", [t0s, tfs, x0s, xfs], [-xfs[0]])  # Maximize final radius[cite: 8]

    e2 = xfs[2] - 0.5  # Final mass constraint m(tf) = 0.5[cite: 8]
    ocp_data2["e"] = cs.Function("e2", [t0s, tfs, x0s, xfs], [e2])
    ocp_data2["e_lb"] = cs.DM([0.0])
    ocp_data2["e_ub"] = cs.DM([0.0])

    ocp_data2["f"] = f_func
    ocp_data2["method"] = method
    ocp_data2["u_lb"] = cs.DM([0.0])  # Coasting T = 0[cite: 8]
    ocp_data2["u_ub"] = cs.DM([0.0])
    ocp_data2["tf_lb"] = cs.DM([0.15])
    ocp_data2["tf_ub"] = cs.DM([0.35])
    ocp_data2["n"] = n_intervals

    ocp_data2['J_s'] = cs.DM([1.0])
    ocp_data2['t_s'] = cs.DM([0.3])
    ocp_data2['x_s'] = cs.DM([1.0, 0.1, 1.0])
    ocp_data2['u_s'] = cs.DM([1.0])
    ocp_data2['e_s'] = cs.DM([1.0])
    ocp_data2['h_s'] = None
    ocp2 = Ocp(**ocp_data2)

    # ----------------------------------------------------
    # 3. NLP Assembly & Phase Linkages (Hard Knots)[cite: 8]
    # ----------------------------------------------------
    nlp = cs.Opti()

    dis_ocp0 = generate_NLP(ocp0, nlp)
    dis_ocp1 = generate_NLP(ocp1, nlp)
    dis_ocp2 = generate_NLP(ocp2, nlp)

    # Link 0: Phase 0 -> Phase 1 (Stage Drop & Hard Knot Conditions)[cite: 8]
    tf0 = dis_ocp0['tf']
    xf0 = dis_ocp0['X'][:, -1]
    t01 = dis_ocp1['t0']
    x01 = dis_ocp1['X'][:, 0]

    nlp.subject_to(tf0 - t01 == 0)
    nlp.subject_to(xf0[0] - x01[0] == 0)  # Radius continuity[cite: 8]
    nlp.subject_to(xf0[1] - x01[1] == 0)  # Velocity continuity[cite: 8]
    nlp.subject_to(x01[2] == 0.7)         # Post-drop mass m^+ = 0.7[cite: 8]
    nlp.subject_to(xf0[2] - x01[2] >= 0.1) # Min structural mass drop[cite: 8]
    nlp.subject_to(xf0[2] - x01[2] <= 0.3) # Max structural mass drop[cite: 8]

    # Link 1: Phase 1 -> Phase 2 (Time & State Continuity)
    tf1 = dis_ocp1['tf']
    xf1 = dis_ocp1['X'][:, -1]
    t02 = dis_ocp2['t0']
    x02 = dis_ocp2['X'][:, 0]

    nlp.subject_to(tf1 - t02 == 0)
    nlp.subject_to(xf1 - x02 == 0)

    # ----------------------------------------------------
    # 4. Realistic Initial Guesses & Solving
    # ----------------------------------------------------
    nlp.solver("ipopt")

    # Time guesses matching physical ascent timeline (~0.28 total units)[cite: 8]
    nlp.set_initial(dis_ocp0["tf"], 0.08)
    nlp.set_initial(dis_ocp1["t0"], 0.08)
    nlp.set_initial(dis_ocp1["tf"], 0.20)
    nlp.set_initial(dis_ocp2["t0"], 0.20)
    nlp.set_initial(dis_ocp2["tf"], 0.28)

    # State profile guesses using linspace
    nlp.set_initial(dis_ocp0["X"], cs.vertcat(
        cs.linspace(1.0, 1.002, n_intervals + 1).T,
        cs.linspace(0.0, 0.02, n_intervals + 1).T,
        cs.linspace(1.0, 0.8, n_intervals + 1).T
    ))
    nlp.set_initial(dis_ocp0["U"], cs.DM.ones(1, n_intervals).T * 1.167)

    nlp.set_initial(dis_ocp1["X"], cs.vertcat(
        cs.linspace(1.002, 1.01, n_intervals + 1).T,
        cs.linspace(0.02, 0.08, n_intervals + 1).T,
        cs.linspace(0.7, 0.52, n_intervals + 1).T
    ))
    nlp.set_initial(dis_ocp1["U"], cs.DM.ones(1, n_intervals ).T * 0.5)

    nlp.set_initial(dis_ocp2["X"], cs.vertcat(
        cs.linspace(1.01, 1.011, n_intervals + 1).T,
        cs.linspace(0.08, 0.0, n_intervals + 1).T,
        cs.DM.ones(1, n_intervals + 1) * 0.5
    ))
    nlp.set_initial(dis_ocp2["U"], cs.DM.zeros(1, n_intervals ).T)

    sol = nlp.solve()

    # ----------------------------------------------------
    # 5. Post-Processing & Verification (with unique names)
    # ----------------------------------------------------
    plot_solution(sol, dis_ocp0, name="Phase 0")
    plot_solution(sol, dis_ocp1, name="Phase 1")
    plot_solution(sol, dis_ocp2, name="Phase 2")
    
    check_feasibility(ocp0, dis_ocp0, sol, name="Phase 0")
    check_feasibility(ocp1, dis_ocp1, sol, name="Phase 1")
    check_feasibility(ocp2, dis_ocp2, sol, name="Phase 2")

if __name__ == "__main__":
    solve_multiphase_ocp()