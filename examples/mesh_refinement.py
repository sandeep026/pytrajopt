import casadi as cs
from pytrajopt import (
    casadi_integrator,
    plot_solution,
    Ocp,
    check_feasibility,
    generate_NLP,
    disc_opt,
    cubic_spline_interp,
    pc_interp,
)


def get_syms(nx, nu):
    # event and Mayer
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", nx, 1)
    xfs = cs.SX.sym("xfs", nx, 1)

    event = (t0s, tfs, x0s, xfs)

    # Lagrange dynamics and path
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", nx, 1)
    us = cs.SX.sym("us", nu, 1)

    path = (ts, xs, us)

    return dict(zip(["path", "event"], [path, event]))


def get_ocp():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1

    syms = get_syms(ocp_data["nx"], ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms["event"]
    ts, xs, us = syms["path"]

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(100)

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", [t0s, tfs, x0s, xfs], [x0s[-1] - xfs[-1]])

    # event constraints
    e = cs.vertcat(x0s - cs.DM([0, 0, 5e5]), xfs[0] - 1e5)

    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0] * 4)
    ocp_data["e_ub"] = cs.vcat([0] * 4)

    g = 9.81
    alp = 1 / (300 * g)

    # dynamics
    f = cs.vcat([xs[1], us / xs[2] - g, -alp * us])
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    ocp_data["u_lb"] = cs.DM([0.0])
    ocp_data["x_lb"] = cs.DM([-cs.DM.inf(), -cs.DM.inf(), 0])

    ocp_data["J_s"] = cs.DM([3e5])
    ocp_data["x_s"] = cs.DM([1e5, 2e3, 3e5])
    ocp_data["u_s"] = cs.DM([1e8])
    ocp_data["e_s"] = cs.DM([1e5, 2e3, 3e5] + [1e5])

    ocp = Ocp(**ocp_data)
    return ocp


def get_disc(ocp, n, grid):
    disc_data = {}
    syms = get_syms(ocp.nx, ocp.nu)
    disc_data["n"] = n
    disc_data["grid"] = grid
    disc_data["method"] = casadi_integrator(
        plugin="cvodes",
        nx=ocp.nx,
        nu=ocp.nu,
        path=list(syms["path"]),
        f=ocp.f(*list(syms["path"])),
    )
    disc = disc_opt(**disc_data)
    return disc


def rocket():
    ocp_data = {}
    ocp_data["nx"] = 3
    ocp_data["nu"] = 1

    syms = get_syms(ocp_data["nx"], ocp_data["nu"])
    t0s, tfs, x0s, xfs = syms["event"]
    ts, xs, us = syms["path"]

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(100)

    # lagrange and mayer objective
    ocp_data["L"] = None
    ocp_data["E"] = cs.Function("E", [t0s, tfs, x0s, xfs], [x0s[-1] - xfs[-1]])

    # event constraints
    e = cs.vertcat(x0s - cs.DM([0, 0, 5e5]), xfs[0] - 1e5)

    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = cs.vcat([0] * 4)
    ocp_data["e_ub"] = cs.vcat([0] * 4)

    g = 9.81
    alp = 1 / (300 * g)

    # dynamics
    f = cs.vcat([xs[1], us / xs[2] - g, -alp * us])
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    ocp_data["u_lb"] = cs.DM([0.0])
    ocp_data["x_lb"] = cs.DM([-cs.DM.inf(), -cs.DM.inf(), 0])

    ocp_data["J_s"] = cs.DM([3e5])
    ocp_data["x_s"] = cs.DM([1e5, 2e3, 3e5])
    ocp_data["u_s"] = cs.DM([1e8])
    ocp_data["e_s"] = cs.DM([1e5, 2e3, 3e5] + [1e5])

    ocp = Ocp(**ocp_data)

    disc_data = {}
    disc_data["n"] = 149
    disc_data["grid"] = cs.hcat(
        [
            cs.linspace(0, 5 / 100, 100).T,
            cs.linspace(5 / 100 + 5 / 100 * 1 / 100, 1, 50).T,
        ]
    )
    disc_data["method"] = casadi_integrator(
        plugin="collocation",
        plugin_options={
            "interpolation_order": 3,
            "collocation_scheme": "legendre",
            "rootfinder": "kinsol",
            "number_of_finite_elements": 1,
            "simplify": True,
        },
        nx=ocp_data["nx"],
        nu=ocp_data["nu"],
        path=[ts, xs, us],
        f=f,
    )

    disc = disc_opt(**disc_data)

    nlp = cs.Opti()

    dis_ocp = generate_NLP(ocp, nlp, disc)
    nlp.solver("ipopt")

    Xg = cs.vcat(
        [
            cs.linspace(0, 0 * 1e5, disc.n + 1).T,
            cs.linspace(0, 0, disc.n + 1).T,
            cs.linspace(1e5, 1e5, disc.n + 1).T,
        ]
    )

    nlp.set_initial(dis_ocp["X"], Xg)
    nlp.set_initial(dis_ocp["U"], cs.DM.zeros(dis_ocp["U"].shape))
    sol = nlp.solve()
    plot_solution(sol, dis_ocp, "rocket_sol")
    check_feasibility(ocp, dis_ocp, sol, "rocket_feas")


import warnings
import numpy as np
import matplotlib.pyplot as plt
import pywt


def wavelet_refined_mesh(
    t,
    U,
    wavelet="haar",
    level=None,
    rel_threshold=0.25,
    samples_per_interval=8,
    max_wavelet_samples=65536,
    max_new_points=20,
    min_interval=1e-6,
    neighbor_intervals=0,
    flat_tol=1e-12,
    return_info=True,
):
    """Wavelet-based refinement for U.shape == (nu, n), t.shape == (n+1,)."""

    t = np.asarray(t, dtype=float)
    U = np.asarray(U, dtype=float)

    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1D array with at least 2 nodes.")
    if U.ndim != 2 or U.shape[1] != len(t) - 1:
        raise ValueError("U must have shape (nu, len(t)-1).")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(U)):
        raise ValueError("Inputs must be finite.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing.")
    if not 0 < rel_threshold <= 1:
        raise ValueError("rel_threshold must lie in (0, 1].")
    if max_new_points is not None and max_new_points < 0:
        raise ValueError("max_new_points must be nonnegative.")

    # Normalize the time horizon to [0, 1].
    t = (t - t[0]) / (t[-1] - t[0])
    t[0], t[-1] = 0.0, 1.0

    n = len(t) - 1
    h = np.diff(t)

    # Uniform auxiliary sampling for wavelet analysis.
    required = max(
        32,
        samples_per_interval * n,
        int(np.ceil(2.0 / np.min(h))),
    )
    m = min(required, max_wavelet_samples)

    if m < required:
        warnings.warn("Auxiliary sampling cap reached.")

    sample_t = (np.arange(m) + 0.5) / m
    parent = np.clip(np.searchsorted(t, sample_t, side="right") - 1, 0, n - 1)

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(m, wav.dec_len)
    used_level = max_level if level is None else min(level, max_level)

    score = np.zeros(n)

    if used_level >= 1:
        for row in U:
            amplitude = np.ptp(row)
            magnitude = np.max(np.abs(row))

            if amplitude <= flat_tol * max(magnitude, 1.0):
                continue

            signal = (row[parent] - np.mean(row)) / amplitude
            coeffs = pywt.wavedec(signal, wav, mode="symmetric", level=used_level)

            activity_sq = np.zeros(m)

            # Reconstruct individual wavelet-detail scales.
            for j in range(1, len(coeffs)):
                parts = [np.zeros_like(c) for c in coeffs]
                parts[j] = coeffs[j]
                detail = pywt.waverec(parts, wav, mode="symmetric")
                detail = detail[:m]
                activity_sq += detail**2

            activity = np.sqrt(activity_sq)
            peak = activity.max()

            if peak > 1e-14:
                activity /= peak
                local = np.zeros(n)
                np.maximum.at(local, parent, activity)
                score = np.maximum(score, local)

    marked = score >= rel_threshold

    if neighbor_intervals > 0:
        original = marked.copy()
        expanded = marked.copy()
        for d in range(1, neighbor_intervals + 1):
            expanded[d:] |= original[:-d]
            expanded[:-d] |= original[d:]
        marked = expanded

    candidates = np.flatnonzero(marked & (h >= 2 * min_interval))

    if max_new_points is not None and len(candidates) > max_new_points:
        order = np.argsort(score[candidates])[::-1]
        candidates = candidates[order[:max_new_points]]

    candidates = np.sort(candidates)
    inserted = 0.5 * (t[candidates] + t[candidates + 1])
    t_new = np.sort(np.concatenate((t, inserted)))
    t_new[0], t_new[-1] = 0.0, 1.0

    info = {
        "interval_scores": score,
        "marked_intervals": np.flatnonzero(marked),
        "refined_intervals": candidates,
        "inserted_nodes": inserted,
        "auxiliary_samples": m,
        "wavelet_level": used_level,
    }

    return (t_new, info) if return_info else t_new


def mesh_refinement(ocp,n_init: int = 10, max_iter: int = 50, n_max=1e4):
    obj = []
    warm = {
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-9,
        "ipopt.warm_start_bound_frac": 1e-9,
        "ipopt.warm_start_slack_bound_push": 1e-9,
        "ipopt.warm_start_mult_bound_push": 1e-9,
        "ipopt.mu_init": 1e-6,
        "ipopt.max_iter": 500,
    }
    for i in range(max_iter):
        if i == 0:
            n = n_init
            grid = cs.linspace(0, 1, n + 1).T
        else:
            t, _ = wavelet_refined_mesh(
                t_temp,
                u_temp.reshape(1, -1),
                wavelet="haar",
                rel_threshold=0.25,
                max_new_points=20,
                return_info=True,
            )

            n = t.size - 1
            grid = cs.DM((t - t[0]) / (t[-1] - t[0])).T

        disc = get_disc(ocp, n, grid)

        nlp = cs.Opti()
        dis_ocp = generate_NLP(ocp, nlp, disc)
        nlp.solver("ipopt", {"ipopt.print_level": 0, "print_time": False} | warm)
        if i == 0:
            Xg = cs.vcat(
                [
                    cs.linspace(0, 0 * 1e5, disc.n + 1).T,
                    cs.linspace(0, 0, disc.n + 1).T,
                    cs.linspace(1e5, 1e5, disc.n + 1).T,
                ]
            )
            Ug = cs.DM.zeros(dis_ocp["U"].shape)
        else:
            td = (t_temp[0] + (t_temp[-1] - t_temp[0]) * disc.grid).full().ravel()
            Xg = cubic_spline_interp(t_temp, x_temp)(td)
            Ug = pc_interp(t_temp, sol.value(u_temp))(td)[0:-1]

        nlp.set_initial(dis_ocp["X"], Xg)
        nlp.set_initial(dis_ocp["U"], Ug)
        sol = nlp.solve()

        t_temp = sol.value(dis_ocp["t"]).ravel()
        x_temp = sol.value(dis_ocp["X"])
        u_temp = sol.value(dis_ocp["U"])

        obj.append(sol.value(nlp.f))

        if i != 0:
            rel = cs.fabs(obj[-1] - obj[-2]) / cs.fabs(obj[-1])
            print(f"{i} | {n} | {rel}  | {sol.stats()['iter_count']}")
            if rel < 1e-3:
                break
            if disc.n > n_max:
                break

    plot_solution(sol, dis_ocp, "rocket_sol")
    check_feasibility(ocp, dis_ocp, sol, "rocket_feas")


if __name__ == "__main__":
    ocp=get_ocp()
    mesh_refinement(ocp)
