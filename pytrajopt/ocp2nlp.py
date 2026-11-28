import casadi as cs
import numpy as np
from pydantic import (
    ConfigDict,
    InstanceOf,
    validate_call,
)

from .costate_estimate import get_costate
from .ocp import DisOcp, Ocp
from .transcription import DynamicMesh, Transcription


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def generate_NLP(
    ocp: InstanceOf[Ocp],
    nlp: cs.Opti,
    transcription: InstanceOf[Transcription],
    mesh: InstanceOf[DynamicMesh] | None = None,
):
    """
    Given the OCP and transcription methods, this function
    generates the discrete OCP and associated opti()
    object
    """
    n = transcription.n
    nx, nu = ocp.nx, ocp.nu
    constraint = {}
    if mesh is None:
        if ocp.t0 is None:
            t0 = nlp.variable()
            if ocp.t_s is not None and ocp.t_o is not None:
                nlp.set_linear_scale(t0, ocp.t_s, ocp.t_o)
        else:
            t0 = ocp.t0
        if ocp.tf is None:
            tf = nlp.variable()
            if ocp.t_s is not None and ocp.t_o is not None:
                nlp.set_linear_scale(tf, ocp.t_s, ocp.t_o)
        else:
            tf = ocp.tf
        if transcription.grid is not None:
            t = t0 + (tf - t0) * transcription.grid
    else:
        t = nlp.variable(1, transcription.n + 1)
        nlp.set_linear_scale(
            t,
            cs.repmat(mesh.scale, 1, transcription.n + 1),
            cs.repmat(mesh.offset, 1, transcription.n + 1),
        )
        t0 = t[:, 0]
        tf = t[:, -1]
        if ocp.t0 is not None:
            constraint["dynamic_mesh_t0"] = (
                nlp.bounded(cs.DM(0), t[:, 0] - ocp.t0, cs.DM(0)),
                ocp.t_s,
            )
            nlp.subject_to(constraint["dynamic_mesh_t0"][0])
        if ocp.tf is not None:
            constraint["dynamic_mesh_tf"] = (
                nlp.bounded(cs.DM(0), t[:, -1] - ocp.tf, cs.DM(0)),
                ocp.t_s,
            )
            nlp.subject_to(constraint["dynamic_mesh_tf"][0])
        dt = cs.diff(t)
        dtn = (tf - t0) / transcription.n
        constraint["dynamics_mesh_dt"] = (
            nlp.bounded(mesh.k1 * dtn, dt, mesh.k2 * dtn),
            cs.repmat(ocp.t_s, 1, n + 1),
        )
        nlp.subject_to(constraint["dynamics_mesh_dt"][0])

    W = nlp.variable(nx + nu, n)
    Xf = nlp.variable(nx, 1)
    X = cs.horzcat(W[0:nx, :], Xf)
    if ocp.x_s is not None and ocp.x_o is not None:
        nlp.set_linear_scale(
            X, cs.repmat(ocp.x_s, 1, n + 1), cs.repmat(ocp.x_o, 1, n + 1)
        )
    U = W[nx:, :]
    if ocp.u_s is not None and ocp.u_o is not None:
        nlp.set_linear_scale(U, cs.repmat(ocp.u_s, 1, n), cs.repmat(ocp.u_o, 1, n))

    # X = nlp.variable(nx, n + 1)
    # nlp.set_linear_scale(X, cs.repmat(ocp.x_s, 1, n + 1), cs.repmat(ocp.x_o, 1, n + 1))
    # U = nlp.variable(nu, n)
    # nlp.set_linear_scale(U, cs.repmat(ocp.u_s, 1, n), cs.repmat(ocp.u_o, 1, n))
    event_ip = (t[:, 0], t[:, -1], X[:, 0], X[:, -1])
    # objective
    if ocp.L is None:
        L = 0
    else:
        L = cs.sum2(
            (t[:, 1:] - t[:, 0:-1]) * ocp.L.map(n, "serial")(t[:, 0:-1], X[:, 0:-1], U)
        )
    if ocp.E is None:
        E = 0
    else:
        E = ocp.E(*event_ip)
    objective = L + E
    nlp.minimize(objective, float(ocp.J_s))
    # event constraints
    e = ocp.e(*event_ip)
    constraint["e"] = (nlp.bounded(ocp.e_lb, e, ocp.e_ub), ocp.e_s)
    nlp.subject_to(*constraint["e"])
    # dynamics
    defect = transcription.get_defect().map(n, "serial")
    f = defect(t[:, 0:-1], t[:, 1:], X[:, 0:-1], X[:, 1:], U)
    zr = cs.DM.zeros(f.shape)
    constraint["f"] = (nlp.bounded(zr, f, zr), cs.repmat(ocp.x_s, 1, n))
    nlp.subject_to(*constraint["f"])
    # path constriants
    if ocp.h is not None:
        # ZOH to extend control at n+1
        h = ocp.h.map(n + 1, "serial")(t, X, cs.hcat([U, U[:, -1]]))
        constraint["h"] = (
            nlp.bounded(
                cs.repmat(ocp.h_lb, 1, n + 1), h, cs.repmat(ocp.h_ub, 1, n + 1)
            ),
            cs.repmat(ocp.h_s, 1, n + 1),
        )
        nlp.subject_to(*constraint["h"])

    # simple bounds
    if ocp.t0 is None:
        constraint["t0"] = (nlp.bounded(ocp.t0_lb, t0, ocp.t0_ub), ocp.t_s)
        nlp.subject_to(*constraint["t0"])

    if ocp.tf is None:
        constraint["tf"] = (nlp.bounded(ocp.tf_lb, tf, ocp.tf_ub), ocp.t_s)
        nlp.subject_to(*constraint["tf"])

    constraint["X"] = (
        nlp.bounded(cs.repmat(ocp.x_lb, 1, n + 1), X, cs.repmat(ocp.x_ub, 1, n + 1)),
        cs.repmat(ocp.x_s, 1, n + 1),
    )
    nlp.subject_to(*constraint["X"])

    constraint["U"] = (
        nlp.bounded(cs.repmat(ocp.u_lb, 1, n), U, cs.repmat(ocp.u_ub, 1, n)),
        cs.repmat(ocp.u_s, 1, n),
    )
    nlp.subject_to(*constraint["U"])

    dis_ocp = {}
    dis_ocp["n"] = n
    dis_ocp["nx"] = ocp.nx
    dis_ocp["nu"] = ocp.nu
    dis_ocp["ne"] = ocp.e.size_out(0)[0]
    dis_ocp["nh"] = ocp.h.size_out(0)[0] if ocp.h is not None else None

    dis_ocp["t"] = t
    dis_ocp["t0"] = t0
    dis_ocp["tf"] = tf
    dis_ocp["X"] = X
    dis_ocp["U"] = U
    dis_ocp["L"] = L
    if ocp.h is None:
        dis_ocp["h"] = None
    else:
        dis_ocp["h"] = h
    dis_ocp["E"] = E
    dis_ocp["e"] = e
    dis_ocp["f"] = ocp.f.map(n + 1, "serial")(t, X, cs.hcat([U, U[:, -1]]))
    dis_ocp["J"] = objective
    if (
        transcription.integrator.plugin in ["bw_euler", "her_sim", "trap"]
        or mesh is not None
    ):
        print("Costate estimation not implemented for IRK methods and adaptive mesh")
        duals = get_costate(ocp, nlp, constraint, t)
        for name, value in duals.items():
            if value is not None:
                duals[name] = np.full(shape=value.shape, fill_value=np.nan)
        dis_ocp.update({"dual": duals})
    else:
        duals = get_costate(ocp, nlp, constraint, t)
        dis_ocp.update({"dual": duals})

    return DisOcp(**dis_ocp)
