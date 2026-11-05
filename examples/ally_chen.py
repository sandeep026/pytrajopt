import casadi as cs

from pytrajopt import (
    Ocp,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
    Syms,
    LinGuess
)

def meta():
    data=r'hen, Weifeng, and Lorenz T. Biegler. “Nested Direct Transcription Optimization for Singular Optimal Control Problems.” AIChE Journal, vol. 62, no. 10, Oct. 2016, pp. 3611–27. DOI.org (Crossref), https://doi.org/10.1002/aic.15272.'
    return data

def ally():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1
    syms=Syms(nx=ocp_data['nx'],nu=ocp_data['nu'])
    t0s,tfs,x0s,xfs=syms.get_event()
    ts,xs,us=syms.get_path()
    ocp_data["L"] = None
    ocp_data["E"] = syms.get_function('E',xfs[2],'event')
    ocp_data["e"] = syms.get_function('e',cs.vertcat(x0s - cs.DM([0,1, 0])),'event')
    ocp_data["e_lb"] = [0] * 3
    ocp_data["e_ub"] = [0] * 3
    ocp_data["t0"] = 0
    ocp_data["tf"] = cs.pi/2
    ocp_data["f"] = syms.get_function('f',cs.vertcat(xs[1],us,0.5*(xs[1]**2-xs[0]**2)),'path')

    inf=cs.DM.inf()
    ocp_data["x_lb"] = [-inf]*3
    ocp_data["x_ub"] = [ inf]*3
    ocp_data["u_lb"] = [-1]
    ocp_data["u_ub"] = [ 1]

    ocp = Ocp(**ocp_data)
    nlp = cs.Opti()

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
    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")

    guess_data = {
        "n": disc.n,
        "xl": [0, 1, 0],
        "xr": [0, 1, 0],
        "ul": [-1],
        "ur": [1],
    }
    guess=LinGuess(**guess_data)
    guess.set_guess(dis_ocp,nlp)
    
    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "ally_chen_sol")
    check_feasibility(ocp, dis_ocp, sol, "ally_chen_feas")

if __name__=='__main__':
    ally()