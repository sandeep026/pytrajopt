from typing import Literal

import casadi as cs
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

dis_methods = Literal["fw_euler", "bw_euler", "trap", "her_sim"]


class TrajOpt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    nx: PositiveInt
    nu: PositiveInt
    n: PositiveInt
    t0: cs.DM | None = None  # none means free var
    tf: cs.DM | None = None  # none means free var
    L: cs.Function | None = None  # none means zero
    f: cs.Function
    h: cs.Function | None = None  # optional
    E: cs.Function | None = None  # zero
    e: cs.Function
    method: dis_methods
    h_lb: cs.DM | None = None  # optional
    h_ub: cs.DM | None = None  # optional
    e_lb: cs.DM
    e_ub: cs.DM
    x_lb: cs.DM | None = None  # optional
    x_ub: cs.DM | None = None  # optional
    u_lb: cs.DM | None = None  # optional
    u_ub: cs.DM | None = None  # optional
    t0_lb: cs.DM | None = None  # optional
    t0_ub: cs.DM | None = None  # optional
    tf_lb: cs.DM | None = None  # optional
    tf_ub: cs.DM | None = None  # optional

    

    @field_validator("f", "L", "h")
    @classmethod
    def path_func_validate(cls, f, info: ValidationInfo):

        if f is None and info.field_name == "f":
            raise ValueError("Mandatory to set dynamics constraint (f)")

        if f is not None:
            if f.n_in() != 3:
                raise ValueError(
                    f"Expected (t,x,u) for {f.name()} as input but received only ({f.n_in()}) inputs"
                )

            if f.n_out() != 1:
                raise ValueError(
                    f"Expected a single output for {f.name()} but received({f.n_out()})"
                )

            if f.size_out(0)[1] != 1:
                raise ValueError(
                    f"functions passed must return column vector but received {f.size_out(0)}"
                )

            inputs = ["t", "x", "u"]
            for i, j in enumerate(
                [(1, 1), (info.data.get("nx"), 1), (info.data.get("nu"), 1)]
            ):
                if f.size_in(i) != j:
                    raise ValueError(
                        f"Dimension of {inputs[i]} must be {j} but received{f.size_in(i)}"
                    )

            if info.field_name == "f" and f.size_out(0) != (info.data.get("nx"), 1):
                raise ValueError(
                    f"dynamics function ({f.name()}) must return a {(info.data.get('nx'), 1)} vector but received {f.size_out(0)}"
                )

            if info.field_name == "L" and f.size_out(0) != (1, 1):
                raise ValueError(
                    f"Lagrange function ({f.name()}) must return a scalar vector but received {f.size_out(0)}"
                )

        return f

    @field_validator("E", "e")
    @classmethod
    def event_func_validate(cls, f, info: ValidationInfo):

        if f is None and info.field_name == "e":
            raise ValueError("Mandatory to set event constraint (e)")

        if f is not None:
            if f.n_in() != 4:
                raise ValueError(
                    f"Expected (t0,tf,x0,xf) for {f.name()} as input but received only ({f.n_in()}) inputs"
                )

            if f.n_out() != 1:
                raise ValueError(
                    f"Expected a single output for {f.name()} but received({f.n_out()})"
                )

            if f.size_out(0)[1] != 1:
                raise ValueError(
                    f"functions passed must return column vector but received {f.size_out(0)}"
                )

            inputs = ["t0", "tf", "x0", "xf"]
            for i, j in enumerate(
                [(1, 1), (1, 1), (info.data.get("nx"), 1), (info.data.get("nx"), 1)]
            ):
                if f.size_in(i) != j:
                    raise ValueError(
                        f"Dimension of {inputs[i]} must be {j} but received{f.size_in(i)}"
                    )

            if info.field_name == "E" and f.size_out(0) != (1, 1):
                raise ValueError(
                    f"Mayer function ({f.name()}) must return a scalar vector but received {f.size_out(0)}"
                )

        return f

    @field_validator(
        "h_lb",
        "h_ub",
        "e_lb",
        "e_ub",
        "x_lb",
        "x_ub",
        "u_lb",
        "u_ub",
        "t0_lb",
        "t0_ub",
        "tf_lb",
        "tf_ub",
    )
    @classmethod
    def shape_validate(cls, vec, info: ValidationInfo):

        if info.data.get("t0") is not None and info.field_name in ["t0_lb", "t0_ub"]:
            return None

        if info.data.get("tf") is not None and info.field_name in ["tf_lb", "tf_ub"]:
            return None

        if info.data.get("h") is None and info.field_name in ["h_lb", "h_ub"]:
            return None

        if info.data.get("e") is None and info.field_name in ["e_lb", "e_ub"]:
            return None

        shapes = {
            "x_lb": (info.data.get("nx"), 1),
            "x_ub": (info.data.get("nx"), 1),
            "u_lb": (info.data.get("nu"), 1),
            "u_ub": (info.data.get("nu"), 1),
            "e_lb": info.data.get("e").size_out(0),
            "e_ub": info.data.get("e").size_out(0),
            "t0_lb": (1, 1),
            "t0_ub": (1, 1),
            "tf_lb": (1, 1),
            "tf_ub": (1, 1),
        }

        if info.data.get("h") is not None:
            shapes.update({"h_lb": info.data.get("h").size_out(0)})
            shapes.update({"h_ub": info.data.get("h").size_out(0)})

        name = info.field_name
        if vec is not None:
            if np.isnan(vec.full()).any():
                raise ValueError(f"{info.field_name} cannot have NaN")

            if vec.shape != shapes[name]:
                raise ValueError(f"{name}: expected {shapes[name]}, got {vec.shape}")
            return vec
        else:
            c = -1 if "_lb" in name else 1
            if "t" in name and "_lb" in name:
                vec = cs.DM(1e-8)
                return vec
            else:
                vec = cs.vcat([c * cs.DM.inf()] * shapes[name][0])
                return vec

    @model_validator(mode="after")
    def validate_trajopt(self):

        if np.any(self.x_lb.full() > self.x_ub.full()):
            raise ValueError("lower must be <= upper bound for x")

        if np.any(self.u_lb.full() > self.u_ub.full()):
            raise ValueError("lower must be <= upper bound for u")

        if self.h is not None:
            if np.any(self.h_lb.full() > self.h_ub.full()):
                raise ValueError("lower must be <= upper bound for h")

        if np.any(self.e_lb.full() > self.e_ub.full()):
            raise ValueError("lower must be <= upper bound for e")

        if self.t0 is None:
            if np.any(self.t0_lb.full() > self.t0_ub.full()):
                raise ValueError("lower must be <= upper bound for t0")

        if self.tf is None:
            if np.any(self.tf_lb.full() > self.tf_ub.full()):
                raise ValueError("lower must be <= upper bound for tf")

        if self.t0 is not None and self.tf is not None:
            if self.t0 > self.tf:
                raise ValueError(
                    f"final time must be greater than inital time but t0 ({self.t0})> tf ({self.tf})"
                )

        return self

    def generate_NLP(self):

        nlp = cs.Opti()

        n = self.n
        nx, nu = self.nx, self.nu

        dec_var = []

        if self.t0 is None:
            t0 = nlp.variable()
            dec_var.append(t0)
        else:
            t0 = self.t0

        if self.tf is None:
            tf = nlp.variable()
            dec_var.append(tf)
        else:
            tf = self.tf

        t = cs.linspace(t0, tf, n + 1).T

        X = nlp.variable(nx, n + 1)
        dec_var.append(X)
        U = nlp.variable(nu, n)
        dec_var.append(U)

        # decision vector
        W = cs.vcat([cs.vec(i) for i in dec_var])

        event_ip = (t[:, 0], t[:, -1], X[:, 0], X[:, -1])

        # objective
        if self.L is None:
            L = 0
        else:
            L = (
                (tf - t0)
                / n
                * cs.sum2(self.L.map(n, "serial")(t[:, 0:-1], X[:, 0:-1], U))
            )
        if self.E is None:
            E = 0
        else:
            E = self.E(*event_ip)
        objective = L + E

        cons = []
        cons_lb = []
        cons_ub = []
        # event constraints
        e = self.e(*event_ip)
        cons.append(e)
        cons_lb.append(self.e_lb)
        cons_ub.append(self.e_ub)

        # dynamics
        f = self.get_discrete_dynamics(t, X, U)
        cons.append(f)
        zr = cs.DM.zeros(f.shape)
        cons_lb.append(zr)
        cons_ub.append(zr)

        # path constriants
        if self.h is not None:
            h = self.h.map(n + 1, "serial")(t, X, cs.hcat([U, U[:, -1]]))
            cons.append(h)
            cons_lb.append(cs.repmat(self.h_lb, 1, n + 1))
            cons_ub.append(cs.repmat(self.h_ub, 1, n + 1))

        # constraints
        G = cs.vcat([cs.vec(i) for i in cons])
        G_lb = cs.vcat([cs.vec(i) for i in cons_lb])
        G_ub = cs.vcat([cs.vec(i) for i in cons_ub])

        # simple bounds
        cons_simple_lb = []
        cons_simple_ub = []

        if self.t0 is None:
            t0_lb = self.t0_lb
            t0_ub = self.t0_ub
            cons_simple_lb.append(t0_lb)
            cons_simple_ub.append(t0_ub)

        if self.tf is None:
            tf_lb = self.tf_lb
            tf_ub = self.tf_ub
            cons_simple_lb.append(tf_lb)
            cons_simple_ub.append(tf_ub)

        X_lb = cs.repmat(self.x_lb, 1, n + 1)
        X_ub = cs.repmat(self.x_ub, 1, n + 1)
        cons_simple_lb.append(X_lb)
        cons_simple_ub.append(X_ub)
        U_lb = cs.repmat(self.u_lb, 1, n)
        U_ub = cs.repmat(self.u_ub, 1, n)
        cons_simple_lb.append(U_lb)
        cons_simple_ub.append(U_ub)

        W_lb = cs.vcat([cs.vec(i) for i in cons_simple_lb])
        W_ub = cs.vcat([cs.vec(i) for i in cons_simple_ub])

        # assemble
        nlp.minimize(objective)
        nlp.subject_to(nlp.bounded(G_lb, G, G_ub))
        nlp.subject_to(nlp.bounded(W_lb, W, W_ub))

        trajopt = {}
        trajopt["nlp"] = nlp
        trajopt["t"] = t
        trajopt["t0"] = t0
        trajopt["tf"] = tf
        trajopt["X"] = X
        trajopt["U"] = U
        trajopt["dynamics"] = f
        trajopt["objective"] = objective

        return trajopt

    def get_discrete_dynamics(self, t, X, U) -> cs.MX:
        F = self.f.map(self.n, "serial")
        Xl = X[:, 0:-1]
        Xr = X[:, 1:]
        tl = t[:, 0:-1]
        tr = t[:, 1:]
        h = t[:, 1] - t[:, 0]
        dX = Xr - Xl
        Fl = F(tl, Xl, U)
        Fr = F(tr, Xr, U)
        if self.method == "bw_euler":
            defect = dX - Fr * h
            return defect
        elif self.method == "fw_euler":
            defect = dX - Fl * h
            return defect
        elif self.method == "trap":
            defect = dX - 0.5 * h * (Fl + Fr)
            return defect
        elif self.method == "her_sim":
            Xm = 0.5 * (Xr + Xl) + h / 8 * (Fl - Fr)
            tm = 0.5 * (tr + tl)
            Fm = F(tm, Xm, U)
            defect = dX - h / 6 * (Fl + 4 * Fm + Fr)
            return defect
        else:
            raise ValueError("Wrong option for method")


if __name__ == "__main__":
    # ocp_data
    ocp_data = {}

    ocp_data["nx"] = 6
    ocp_data["nu"] = 2

    # event and Mayer
    t0s = cs.SX.sym("t0s")
    tfs = cs.SX.sym("tfs")
    x0s = cs.SX.sym("x0s", ocp_data["nx"], 1)
    xfs = cs.SX.sym("xfs", ocp_data["nx"], 1)

    # Lagrange dynamics and path
    ts = cs.SX.sym("ts")
    xs = cs.SX.sym("xs", ocp_data["nx"], 1)
    us = cs.SX.sym("us", ocp_data["nu"], 1)
    c1 = 2.83374
    c2 = -0.80865
    c3 = 0.71265
    c4 = 17.2656
    c5 = 27.0756
    rho = 0.01

    # ocp_data={}

    # initial and final time
    ocp_data["t0"] = cs.DM(0)
    ocp_data["tf"] = cs.DM(9)
    ocp_data["t0_lb"] = None
    ocp_data["tf_lb"] = None
    ocp_data["t0_ub"] = None
    ocp_data["tf_ub"] = None

    # lagrange and mayer objective
    ocp_data["L"] = cs.Function(
        "L",
        [ts, xs, us],
        [0.5 * (xs[2] ** 2 + xs[5] ** 2 + rho * (us[0] ** 2 + us[1] ** 2))],
    )
    ocp_data["E"] = None

    # event constraints
    e = cs.vertcat(
        x0s - cs.DM([0, 22, 0, 0, -1, 0]), xfs - cs.DM([10, 14, 0, 2.5, 0, 0])
    )
    zr = cs.DM.zeros(e.shape)
    ocp_data["e"] = cs.Function("e", [t0s, tfs, x0s, xfs], [e])
    ocp_data["e_lb"] = zr
    ocp_data["e_ub"] = zr

    # dynamics
    f = cs.vcat(
        [
            xs[3],
            xs[4],
            xs[5],
            us[0] + c4 * xs[2],
            us[1],
            -(us[0] + c5 * xs[2] + 2 * xs[4] * xs[5]) / xs[1],
        ]
    )
    ocp_data["f"] = cs.Function("fss", [ts, xs, us], [f])

    # path constraint
    ocp_data["h"] = None
    ocp_data["h_lb"] = None
    ocp_data["h_ub"] = None

    # simple bounds
    inf = cs.DM.inf()
    ocp_data["x_lb"] = cs.DM([-inf] * 3 + [-2.5, -1] + [-inf])
    ocp_data["x_ub"] = cs.DM([+inf] * 3 + [2.5, 1] + [inf])
    ocp_data["u_lb"] = cs.DM([-c1, c2])
    ocp_data["u_ub"] = cs.DM([c1, c3])

    ocp_data["n"] = 100
    ocp_data["method"] = "her_sim"

    trajopt = TrajOpt(**ocp_data)

    nlp = trajopt.generate_NLP()
    nlp["nlp"].solver("ipopt", {"detect_simple_bounds": False})
    nlp["nlp"].set_initial(nlp["X"], cs.DM.ones(nlp["X"].shape))
    nlp["nlp"].set_initial(nlp["U"], cs.DM.zeros(nlp["U"].shape))
    nlp["nlp"].solve()
