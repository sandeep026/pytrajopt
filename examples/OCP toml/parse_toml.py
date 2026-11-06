import importlib
import inspect
import pathlib
import tomllib

import casadi as cs

from pytrajopt import (
    DynamicMesh,
    LinGuess,
    Ocp,
    Syms,
    check_feasibility,
    disc_opt,
    generate_NLP,
    plot_solution,
    plot_sparsity,
)

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()


def toml2sol(
    filename: str,
    opti_settings: dict | None = None,
    soln: bool = True,
    feas: bool = True,
    spar: bool = False,
) -> tuple[Ocp, disc_opt, LinGuess]:

    with open(SCRIPT_DIR / (filename + ".toml"), "rb") as f:
        data = tomllib.load(f)
    print("successfully read")

    module = importlib.import_module(data["expressions"]["func_name"])

    syms = Syms(nx=data["ocp"]["nx"], nu=data["ocp"]["nu"])
    t0s, tfs, x0s, xfs = syms.get_event()
    ts, xs, us = syms.get_path()

    exp = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name in ["L", "f", "h"]:
            exp[name] = func(ts, xs, us)
        elif name in ["E", "e"]:
            exp[name] = func(t0s, tfs, x0s, xfs)
        else:
            raise ValueError(
                f"Function names allowed are L,E,e,f,h but received {name}"
            )

    syms.collect_expression(exp)

    ocp = Ocp(**data["ocp"] | syms.generate_funcs())
    disc = disc_opt(**data["transcription"])
    guess = LinGuess(**data["guess"])
    if data["dynamic_mesh"]:
        mesh = DynamicMesh(**data["dynamic_mesh"])
    else:
        mesh = None

    nlp = cs.Opti()
    if opti_settings is None:
        nlp.solver("ipopt")
    else:
        nlp.solver("ipopt", opti_settings)
    dis_ocp = generate_NLP(ocp, nlp, disc, mesh)
    guess.set_guess(dis_ocp, nlp)
    sol = nlp.solve()
    if soln:
        plot_solution(sol, dis_ocp, f"{filename}_sol")
    if feas:
        check_feasibility(ocp, dis_ocp, sol, f"{filename}_feas")
    if spar:
        plot_sparsity(nlp, f"{filename}_spar")

    return ocp, disc, guess
