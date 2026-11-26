import casadi as cs

from pytrajopt import (
    DisOcp,
    Integrator,
    LinGuess,
    Ocp,
    Syms,
    Transcription,
    check_feasibility,
    generate_NLP,
    plot_solution,
)


# time dilation vs variable mesh
def meta():
    data = r"Betts, John T. Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. Second, Society for Industrial and Applied Mathematics, 2010. DOI.org (Crossref), https://doi.org/10.1137/1.9780898718577."
    return data


def get_ocp(aug: bool):
    if not aug:
        ocp_data = {}
        ocp_data["nx"] = 1
        ocp_data["nu"] = 1
        syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
        t0s, tfs, x0s, xfs = syms.get_event()
        ts, xs, us = syms.get_path()
        ocp_data["L"] = syms.get_function("L", 0.5 * (xs**2 + us**2), "path")
        ocp_data["e"] = syms.get_function(
            "e", cs.vertcat(x0s - cs.DM([1.5]), xfs - 1), "event"
        )
        ocp_data["e_lb"] = [0] * 2
        ocp_data["e_ub"] = [0] * 2
        ocp_data["t0"] = 0
        ocp_data["tf"] = 10000
        ocp_data["f"] = syms.get_function("L", -((xs) ** 3) + us, "path")
        ocp = Ocp(**ocp_data)
        path = [ts, xs, us]
        return ocp, path
    else:
        ocp_data = {}
        ocp_data["nx"] = 2
        ocp_data["nu"] = 2
        syms = Syms(nx=ocp_data["nx"], nu=ocp_data["nu"])
        t0s, tfs, x0s, xfs = syms.get_event()
        ts, xs, us = syms.get_path()
        ocp_data["L"] = syms.get_function(
            "L", 0.5 * (xs[0] ** 2 + us[0] ** 2) * us[1], "path"
        )
        ocp_data["e"] = syms.get_function(
            "e", cs.vertcat(x0s - cs.DM([1.5, 0]), xfs - cs.DM([1, 10000])), "event"
        )
        ocp_data["e_lb"] = [0] * 4
        ocp_data["e_ub"] = [0] * 4
        ocp_data["t0"] = 0
        ocp_data["tf"] = 1
        ocp_data["f"] = syms.get_function(
            "L", cs.vertcat((-((xs[0]) ** 3) + us[0]) * us[1], us[1]), "path"
        )
        inf = cs.DM.inf()
        ocp_data["u_lb"] = [-inf, 1e-8]
        ocp_data["u_ub"] = [inf, inf]
        ocp = Ocp(**ocp_data)
        path = [ts, xs, us]
        return ocp, path


def set_guess(
    nlp: cs.Opti, dis_ocp: DisOcp, transcription: Transcription, aug: bool = False
):
    if not aug:
        guess_data = {
            "n": transcription.n,
            "xl": [1.5],
            "xr": [1],
            "ul": [1.5**3],
            "ur": [1**3],
            "tl": 0,
            "tr": 10000,
        }
        guess = LinGuess(**guess_data)
        guess.set_guess(dis_ocp, nlp)

    else:
        guess_data = {
            "n": transcription.n,
            "xl": [1.5, 0],
            "xr": [1, 10000],
            "ul": [1.5**3 - 5e-6, 10000],
            "ur": [1**3 - 5e-6, 10000],
            "tl": 0,
            "tr": 10000,
        }
        guess = LinGuess(**guess_data)
        guess.set_guess(dis_ocp, nlp)


def solve():
    aug = True
    ocp, path = get_ocp(aug)
    nlp = cs.Opti()
    integrator = Integrator(
        plugin="her_sim",
        # plugin_options={
        #    "collocation_scheme": "legendre",
        #    "interpolation_order": 1,
        #    "number_of_finite_elements": 1,
        # },
        nx=ocp.nx,
        nu=ocp.nu,
        path=path,
        f=ocp.f(*path),
    )
    n = 100
    transcription = Transcription(n=n, integrator=integrator)
    from pytrajopt import DynamicMesh

    if aug:
        mesh = None
    else:
        mesh = DynamicMesh(k1=0.000001, k2=10000000)
    dis_ocp = generate_NLP(ocp, nlp, transcription, mesh)
    nlp.solver("ipopt", {"ipopt.tol": 1e-7})

    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "hyper_sol")
    check_feasibility(ocp, dis_ocp, sol, "hyper_feas")


if __name__ == "__main__":
    solve()
