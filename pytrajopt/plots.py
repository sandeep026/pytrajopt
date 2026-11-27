from pathlib import Path

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline

from .ocp import DisOcp, Ocp

CURRENT_DIR = Path(__file__).parent
STYLE_PATH = CURRENT_DIR / "pub_qual.mplstyle"
plt.style.use(STYLE_PATH)

import sys


def _running_script_dir() -> Path:
    """
    current working directory where the user script is located
    """
    script_arg = sys.argv[0] if sys.argv else ""
    if script_arg:
        script_path = Path(script_arg).resolve()
        if script_path.exists():
            return script_path.parent
    return Path.cwd()


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def pc_interp(t: np.ndarray, U: np.ndarray):
    """
    piecewise constant interpolation for controls

    handles 1d and 2d arrays
    """
    if U.ndim == 2:
        U = np.hstack([U, U[:, -1].reshape(-1, 1)])
        axis = 1
    else:
        U = np.hstack([U, U[-1]])
        axis = 0
    zoh = make_interp_spline(t, U, k=0, axis=axis)
    return zoh


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def cubic_spline_interp(t: np.ndarray, X: np.ndarray):
    """
    piece wise cubic spline interpolation

    handles 1d and 2d arrays

    """
    if X.ndim == 2:
        axis = 1
    else:
        axis = 0
    cubic = make_interp_spline(t, X, k=3, axis=axis)
    return cubic


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plotting(t: np.ndarray, var: np.ndarray, name: str, pdf: PdfPages):
    """
    sets up formatting of the plot

    costates ignored for certain discretizations

    step vs line plots for control and state
    """
    fig, ax = plt.subplots()
    for i in range(var.shape[0]):
        ax.clear()
        if np.isnan(var).all():
            print(
                "Costate estimation not implemented for bw_euler, trap and her_sim. Use fw_euler or exp_mid or rk4"
            )
            continue

        if name == "u":
            ax.step(t.ravel(), var[i, :].ravel())
        else:
            ax.plot(t.ravel(), var[i, :].ravel())
        ax.set_ylabel(rf"${name}_{i}$")
        ax.set_xlabel("time [s]")
        ax.grid(True)
        pdf.savefig(fig, bbox_inches="tight")
    plt.close()


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_solution(sol: cs.OptiSol, dis_ocp: DisOcp, name: str = "solution"):
    """
    compute the numerical value of the discrete state,
    control and costates

    process time for state, costate and control

    save the plots to disc as pdf
    """
    tx = sol.value(dis_ocp.t)
    tu = tx[0:-1]
    tlam = tx[1:]
    tvars = [tx, tu, tlam]
    X = sol.value(dis_ocp.X).reshape(dis_ocp.X.shape)
    U = sol.value(dis_ocp.U).reshape(dis_ocp.U.shape)
    lam = sol.value(dis_ocp.dual["lam_f"]).reshape(dis_ocp.dual["lam_f"].shape)

    with PdfPages(_running_script_dir() / f"{name}.pdf") as pdf:
        for name, var, tvar in zip(["x", "u", r"\lambda"], [X, U, lam], tvars):
            plotting(tvar, var, name, pdf=pdf)


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def check_feasibility(
    ocp: Ocp, dis_ocp: DisOcp, sol: cs.OptiSol, name: str = "feasibility"
):
    """
    time march solution from NLP using Radau integrator of
    scipy.integrator.solve_ivp to verfify if time marched
    solution matches the discrete states

    a very low tolerance for integrator used for accuracy

    results are plotted and saved in a pdf

    conversion of casadi function done to be compatible with scipy
    ineterface

    control is approximated as a continuous piecewise constant
    function

    auto diff jacobian provided for Radau integrator

    TODO: return a estimate of the error
    """

    X = sol.value(dis_ocp.X).reshape(ocp.nx, -1)
    U = sol.value(dis_ocp.U).reshape(ocp.nu, -1)
    tg = sol.value(dis_ocp.t)
    x0 = X[:, 0]
    f = ocp.f

    zoh = pc_interp(tg, U)

    def f_solve_ivp(t, x):
        return f(t, x, zoh(t)).full().flatten()

    def f_solve_ivp_jac(t, x):
        return f.jacobian()(t, x, zoh(t), 0)[1].full().reshape(ocp.nx, ocp.nx)

    td = np.linspace(tg[0], tg[-1], len(tg) * 10)
    options = {"rtol": 1e-12, "atol": 1e-12}
    ode_meth = "Radau"
    ivp_sol = solve_ivp(
        f_solve_ivp, [tg[0], tg[-1]], x0, ode_meth, td, options, jac=f_solve_ivp_jac
    )

    if ivp_sol.success is False:
        raise ValueError("Time marching failed")

    fig, ax = plt.subplots()

    with PdfPages(_running_script_dir() / f"{name}.pdf") as pdf:
        for i in range(x0.size):
            ax.clear()
            #            plt.figure()
            if X.ndim == 1:
                ax.plot(tg, X, label="NLP")
                ax.plot(td, ivp_sol.y, label=ode_meth + " IIA (time-marched)")
            else:
                ax.plot(tg, X[i, :], "o", label="NLP")
                ax.plot(
                    td, ivp_sol.y[i, :], "--", label=ode_meth + " IIA (time-marched)"
                )
            ax.set_xlabel("time [s]")
            ax.set_ylabel(rf"$x_{i}$")
            ax.grid(True)
            ax.legend()
            pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        if U.ndim == 1:
            nu = 1
        else:
            nu = U.shape[0]
        Ud = zoh(td)
        for i in range(nu):
            if U.ndim == 1:
                plt.plot(tg[0:-1], U, "o", label="NLP", markeredgecolor="black")
                plt.plot(td, Ud, "--", label="ZOH")
            else:
                plt.plot(tg[0:-1], U[i, :], "o", label="NLP")
                plt.plot(td, Ud[i, :], "--", label="ZOH")
            plt.xlabel("time [s]")
            plt.ylabel(rf"$u_{i}$")
            plt.grid(True)
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_sparsity(
    nlp: cs.Opti, filename: str = "sparsity", raster_threshold: int = 50000
):
    """
    Prints the sparsity pattern of the
    Jacobian and Hessian to a pdf
    """
    x = nlp.x
    f = nlp.f
    g = nlp.g
    lam = nlp.lam_g
    L = f + cs.dot(lam, g)
    J = cs.jacobian(g, x)
    H = cs.hessian(L, x)[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    with PdfPages(f"{filename}.pdf") as pdf:
        for name, mat in zip(["Jacobian", "Hessian"], [J, H]):
            ax.clear()
            rows, cols = mat.sparsity().get_triplet()
            S = sp.coo_matrix(
                (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=mat.shape
            )

            use_raster = mat.nnz() > raster_threshold
            ax.spy(S, marker=".", markersize=1, rasterized=use_raster)
            ax.set_aspect("equal")
            ax.set_title(name)
            pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def plot_solution_all_phase(dis_ocps: list, sol: cs.OptiSol, name: str = "all"):
    """
    If problem is multiphase, plots the states and controls
    of all phases (if shared same states and contros)
    in a single plot
    """
    with PdfPages(f"{name}.pdf") as pdf:
        for j in range(3):
            plt.figure()
            for i, dis_ocp in enumerate(dis_ocps):
                t = sol.value(dis_ocp.t).ravel()
                X = sol.value(dis_ocp.X)
                if X.ndim == 1:
                    plt.plot(t, X, label=f"phase {i}")
                else:
                    plt.plot(t, X[j, :].ravel(), label=f"phase {i}")
            plt.xlabel("time [s]")
            plt.ylabel(f"$x_{j}$")
            plt.grid()
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()

        for j in range(1):
            plt.figure()
            for i, dis_ocp in enumerate(dis_ocps):
                t = sol.value(dis_ocp.t)
                U = sol.value(dis_ocp.U)
                zoh = pc_interp(t, U)
                zoh = zoh(t)
                if zoh.ndim == 1:
                    plt.plot(t, zoh, label=f"phase {i}")
                else:
                    plt.plot(t, zoh[j, :].ravel(), label=f"phase {i}")
            plt.xlabel("time [s]")
            plt.ylabel(f"$u_{j}$")
            plt.grid()
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()
