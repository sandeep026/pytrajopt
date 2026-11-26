import casadi as cs

from .ocp import Ocp


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def kkt_ub_lb(kkt: cs.DM):
    if kkt.shape != (1, 1):
        pieces = kkt.shape[1] // 2
        lbub = cs.horzsplit(kkt, pieces)
    else:
        pass
    return lbub[1] - lbub[0]


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_costate(ocp: Ocp, nlp: cs.Opti, constraints: dict, t: cs.DM | cs.MX):
    dual = {}

    for name, constraint in constraints.items():
        dual[f"kkt_{name}"] = kkt_ub_lb(nlp.dual(constraint[0]))
        if name == "e":
            dual[f"lam_{name}"] = dual[f"kkt_{name}"]
        if name == "f":
            dual[f"lam_{name}"] = -dual[f"kkt_{name}"]
        if name == "h":
            dt = t[:, 1:] - t[:, 0:-1]
            dual[f"lam_{name}"] = dual[f"kkt_{name}"][:, 0:-1] / cs.repmat(
                dt, dual[f"kkt_{name}"].shape[0], 1
            )

    return dual
