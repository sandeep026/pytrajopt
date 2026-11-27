import casadi as cs

from .ocp import Ocp


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def kkt_ub_lb(kkt: cs.DM):
    """
    lb<=g<=ub

    Multiplier

    lb-g<=0

    g-ub<=0

    There will be 2 multipliers for 2 constraints. But for the
    bounded constraint can be merged into 1.

    lam_ub-lam_lb
    """
    if kkt.shape != (1, 1):
        pieces = kkt.shape[1] // 2
        lbub = cs.horzsplit(kkt, pieces)
    else:
        pass
    return lbub[1] - lbub[0]


# @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_costate(ocp: Ocp, nlp: cs.Opti, constraints: dict, t: cs.DM | cs.MX):
    """
    extracts the kkt multipliers for all the
    constraints

    for certain constraints the costates are also
    estimated from the kkt multipliers

    kkt multipliers of the defect constraints are
    costates for states (only for explicit rk method or
    multiple shooting constraints of the form x(k+1)=phi(x(k),u(k))).

    rk4, fw_euler, exp_mid, casadi integrators (implicit and explicit)
    all provide costate estimates
    """

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
