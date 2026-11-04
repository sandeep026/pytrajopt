import casadi as cs
import numpy as np

from pytrajopt import (
    Ocp,
    Syms,
    casadi_integrator,
    check_feasibility,
    disc_opt,
    generate_NLP,
    pc_interp,
    plot_solution,
    plot_solution_all_phase,
    LinGuess,
    plot_sparsity
)

def meta():    
    data=r"Ross, I. Michael, and Fariba Fahroo. Pseudospectral Knotting Methods " \
    "for Solving Nonsmooth Optimal Control Problems. Journal of Guidance, Control, " \
    "and Dynamics, vol. 27, no. 3, May 2004, pp. 397–405. DOI.org (Crossref), " \
    "https://doi.org/10.2514/1.3426."
    return data

def get_times(id):
    if id == 0:
        t0, tf = cs.DM(0), None
        return {"t0": t0, "tf": tf}
    elif id in [1,2]:
        t0, tf = None, None
        return {"t0": t0, "tf": tf}
    else:
        raise ValueError("Incorrect ID")


def get_lagrange(nx, nu, id):
    if id in [0, 1, 2]:
        return {"L": None}
    else:
        raise ValueError("Incorrect ID")


def get_mayer(nx, nu, id):
    sym = Syms(nx=nx, nu=nu)
    t0, tf, x0, xf = sym.get_event()
    event = sym.get_event()
    if id in [0, 1]:
        return {"E": None}
    elif id == 2:
        return {"E": cs.Function("E", event, [-xf[0]])}
    else:
        raise ValueError("Incorrect ID")


def get_dynamics(nx, nu, id):
    sym = Syms(nx=nx, nu=nu)
    t, x, u = sym.get_path()
    path = sym.get_path()
    if id in [0, 1, 2]:
        mu, ve = 1.0, 0.5
        r, v, m = x[0, :], x[1, :], x[2, :]
        f = cs.vertcat(v, -mu / (r.T @ r) + u / m, -u / ve)
        return {"f": cs.Function("f", path, [f])}
    else:
        raise ValueError("Incorrect ID")


def get_event(nx, nu, id):
    sym = Syms(nx=nx, nu=nu)
    t0, tf, x0, xf = sym.get_event()
    event = sym.get_event()
    if id == 0:
        e = cs.vertcat(x0 - cs.DM([1.0, 0.0, 1.0]))
        lb = cs.DM.zeros(e.shape)
        ub = cs.DM.zeros(e.shape)
        return {"e": cs.Function("e", event, [e]), "e_lb": lb, "e_ub": ub}
    elif id == 1:
        e = cs.vertcat(tf - t0, x0[2] - 0.7)
        lb = cs.DM([1e-6, 0])
        ub = cs.DM([100, 0])
        return {"e": cs.Function("e", event, [e]), "e_lb": lb, "e_ub": ub}
    elif id == 2:
        e = cs.vertcat(xf[2] - 0.5, tf - t0)
        lb = cs.DM([0, 1e-6])
        ub = cs.DM([0, 100])
        return {"e": cs.Function("e", event, [e]), "e_lb": lb, "e_ub": ub}
    else:
        raise ValueError("Incorrect ID")


def get_simple(name, id):
    if id == 0:
        if name == "X":
            lb, ub = cs.DM([0] * 3), cs.DM([2] * 3)
            return {"x_lb": lb, "x_ub": ub}
        elif name == "U":
            lb, ub = cs.DM([1.167]), cs.DM([1.167])
            return {"u_lb": lb, "u_ub": ub}
        elif name == "t0":
            lb, ub = None, None
            return {"t0_lb": lb, "t0_ub": ub}
        elif name == "tf":
            lb, ub = None, None
            return {"tf_lb": lb, "tf_ub": ub}
        else:
            raise ValueError(f"Incorrect name {name}")
    elif id == 1:
        if name == "X":
            lb, ub = cs.DM([0] * 3), cs.DM([2] * 3)
            return {"x_lb": lb, "x_ub": ub}
        elif name == "U":
            lb, ub = cs.DM([0]), cs.DM([0.875])
            return {"u_lb": lb, "u_ub": ub}
        elif name == "t0":
            lb, ub = None, None
            return {"t0_lb": lb, "t0_ub": ub}
        elif name == "tf":
            lb, ub = None, None
            return {"tf_lb": lb, "tf_ub": ub}
        else:
            raise ValueError(f"Incorrect name {name}")
    elif id == 2:
        if name == "X":
            lb, ub = cs.DM([0] * 3), cs.DM([2] * 3)
            return {"x_lb": lb, "x_ub": ub}
        elif name == "U":
            lb, ub = cs.DM([0]), cs.DM([0])
            return {"u_lb": lb, "u_ub": ub}
        elif name == "t0":
            lb, ub = None, None
            return {"t0_lb": lb, "t0_ub": ub}
        elif name == "tf":
            lb, ub = None, None
            return {"tf_lb": lb, "tf_ub": ub}
        else:
            raise ValueError(f"Incorrect name {name}")
    else:
        raise ValueError("Incorrect ID")


def get_disc(id) -> disc_opt:
    if id == 0:
        disc_data = {
            "n": 15,
            "method": "exp_mid",
        }
        return disc_opt(**disc_data)
    elif id == 1:
        disc_data = {
            "n": 25,
            "method": "her_sim",
        }
        return disc_opt(**disc_data)
    elif id == 2:
        disc_data = {
            "n": 30,
            "method": "trap",
        }
        return disc_opt(**disc_data)
    else:
        raise ValueError("Incorrect phase ID")


def get_guess(dis_ocp, nlp: cs.Opti, id):
    if id == 0:
        guess_data = {
            "n": get_disc(id).n,
            "xl": [1, 0, 1],
            "xr": [1.002, 0.02, 0.8],
            "ul": [1.167],
            "ur": [1.167],
            "tl": 0,
            "tr": 0.08,
        }
        guess = LinGuess(**guess_data)
        guess.set_guess(dis_ocp, nlp)
    elif id == 1:
        guess_data = {
            "n": get_disc(id).n,
            "xl": [1.01, 0.08, 1],
            "xr": [1.002, 0.0, 1],
            "ul": [0.5],
            "ur": [0.5],
            "tl": 0.08,
            "tr": 0.2,
        }
        guess = LinGuess(**guess_data)
        guess.set_guess(dis_ocp, nlp)
    elif id == 2:
        guess_data = {
            "n": get_disc(id).n,
            "xl": [1.002, 0.02, 0.7],
            "xr": [1.01, 0.08, 0.52],
            "ul": [0],
            "ur": [0],
            "tl": 0.2,
            "tr": 0.28,
        }
        guess = LinGuess(**guess_data)
        guess.set_guess(dis_ocp, nlp)
    else:
        raise ValueError("Incorrect phase ID")


from functools import reduce


def create_multiphase_ocp():
    nx = 3
    nu = 1
    ip = {"nx": nx, "nu": nu}
    nlp = cs.Opti()
    nlp.solver("ipopt")
    dis_ocps = []
    #important to add objectives across phases
    obj = 0
    for i in range(3):
        list_simple = [get_simple(name, i) for name in ["X", "U", "t0", "tf"]]
        ocp_data = {"nx": nx, "nu": nu}
        ocp_data = ocp_data | (
            get_times(i)
            |get_lagrange(**(ip | {"id": i}))
            | get_mayer(**(ip | {"id": i}))
            | get_dynamics(**(ip | {"id": i}))
            | get_event(**(ip | {"id": i}))
            | reduce(lambda x, y: x | y, list_simple)
        )

        ocp = Ocp(**ocp_data)
        dis_ocp = generate_NLP(ocp=ocp, nlp=nlp, disc_opt=get_disc(id=i))
        dis_ocps.append(dis_ocp)
        get_guess(dis_ocp, nlp, i)
        if i != 0:
            ocp_l = dis_ocps[i - 1]
            ocp_r = dis_ocps[i]
            nlp.subject_to(ocp_l.tf - ocp_r.t0 == 0)
            nlp.subject_to(ocp_l.X[0:2, -1] - ocp_r.X[0:2, 0] == 0)
        if i == 1:
            ocp_l = dis_ocps[i - 1]
            ocp_r = dis_ocps[i]
            nlp.subject_to(nlp.bounded(0.1, ocp_l.X[2, -1] - ocp_r.X[2, 0], 0.3))
        if i == 2:
            ocp_l = dis_ocps[i - 1]
            ocp_r = dis_ocps[i]
            nlp.subject_to(ocp_l.X[2, -1] - ocp_r.X[2, 0] == 0)
        obj = obj + dis_ocp.J

    nlp.minimize(obj)
    sol = nlp.solve()

    plot_solution_all_phase(dis_ocps, sol, "rocket_multiphase")
    plot_sparsity(nlp,'rocket_multiphase_sparsity')


if __name__ == "__main__":
    create_multiphase_ocp()
