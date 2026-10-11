from typing import Literal

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)


class casadi_integrator(BaseModel):
    # provides access to casadi integrators
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    plugin: Literal["cvodes", "collocation", "rk"] = "rk"
    plugin_options: dict = Field(default_factory=dict)
    nx: int
    nu: int
    path: list[cs.SX]
    f: cs.SX
    _scaled_ode: dict[str, cs.SX] = PrivateAttr(default_factory=dict)

    @field_validator("path")
    @classmethod
    def check_length(cls, path):
        if len(path) != 3:
            raise ValueError(
                f"t,x,u must be present in path. Recived {len(path)} expected 3"
            )
        return path

    @field_validator("f")
    @classmethod
    def check_column(cls, f):
        if f.shape[1] != 1:
            raise ValueError(
                f"f must be a column vector but received f of shape {f.shape}"
            )
        return f

    def __eq__(self, other):
        # If the other object is a string, check if it matches the class name
        if isinstance(other, str):
            return self.__class__.__name__ == other

        # Fallback to standard comparison for other object types
        return super().__eq__(other)

    @classmethod
    def get_scaled_dynamics(cls, path, f):
        t0s = cs.SX.sym("t0s")
        tfs = cs.SX.sym("tfs")
        tau = cs.SX.sym("tau")
        t = t0s + (tfs - t0s) * tau
        f_scaled = cs.transform((tfs - t0s) * cs.substitute(f, path[0], t))
        ode_s = {
            "t": tau,
            "x": path[1],
            "p": cs.vertcat(path[2], t0s, tfs),
            "ode": f_scaled,
        }
        return ode_s

    @model_validator(mode="after")
    def generate_ode_dict(self):

        syms_shapes = [(1, 1), (self.nx, 1), (self.nu, 1), (self.nx, 1)]
        syms = self.path + [self.f]
        for name, known_shape, object in zip(["t", "x", "u", "f"], syms_shapes, syms):
            if object.shape != known_shape:
                raise ValueError(
                    f"{name} is a {known_shape} vector but received {object.shape}"
                )

        self._scaled_ode = casadi_integrator.get_scaled_dynamics(self.path, self.f)

        return self


# fw_euler - forward Euler
# bw_euler - backward Euler
# trap     - trapezoidal
# her_sim  - Hermite Simpson
dis_methods = (
    Literal["fw_euler", "bw_euler", "exp_mid", "trap", "her_sim"] | casadi_integrator
)


class TrajOpt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    # dimensions of state control and discretization intervals
    nx: PositiveInt
    nu: PositiveInt
    n: PositiveInt
    # initial and final time
    t0: cs.DM | None = None  # none means free var
    tf: cs.DM | None = None  # none means free var
    # Lagrange (L), Mayer (E), event (e), dynamics (f), path (h) functions
    L: cs.Function | None = None  # none means zero
    E: cs.Function | None = None  # zero
    f: cs.Function
    e: cs.Function
    h: cs.Function | None = None  # optional
    # numerical method for dynamics approximation
    method: dis_methods
    # path constraint bounds
    h_lb: cs.DM | None = None  # optional
    h_ub: cs.DM | None = None  # optional
    # event constraint bounds
    e_lb: cs.DM
    e_ub: cs.DM
    # simple bounds on state and control
    x_lb: cs.DM | None = None  # optional
    x_ub: cs.DM | None = None  # optional
    u_lb: cs.DM | None = None  # optional
    u_ub: cs.DM | None = None  # optional
    # bounds in initial and final time if free
    t0_lb: cs.DM | None = None  # optional
    t0_ub: cs.DM | None = None  # optional
    tf_lb: cs.DM | None = None  # optional
    tf_ub: cs.DM | None = None  # optional

    @field_validator("f", "L", "h")
    @classmethod
    def path_func_validate(cls, f: cs.Function) -> cs.Function:
        """

        Check input and output structure of path function

        f(t,x,u) -> (,1)


        Parameters
        ----------
        f : cs.Function |None


        Returns
        -------
        cs.Function|none

        Raises
        ------
        ValueError
            number of inputs mismatch
        ValueError
            number of outputs mismatch
        ValueError
            output is not a column vector
        """

        if f is not None:
            if f.n_in() != 3:
                raise ValueError(
                    f"Expected (t,x,u) for {f.name()} as input but received only ({f.n_in()}) inputs"
                )

            if f.n_out() != 1:
                raise ValueError(
                    f"Expected a single output for {f.name()} but received ({f.n_out()})"
                )

            if f.size_out(0)[1] != 1:
                raise ValueError(
                    f"functions passed must return column vector but received {f.size_out(0)}"
                )

        return f

    @field_validator("E", "e")
    @classmethod
    def event_func_validate(cls, f: cs.Function | None) -> cs.Function | None:
        """

        Check input and output structure of event function

        f(t0,tf,x0,xf) -> (,1)

        Parameters
        ----------
        f : cs.Function|None


        Returns
        -------
        cs.Function|None

        Raises
        ------
        ValueError
            number of inputs mismatch
        ValueError
            number of outputs mismatch
        ValueError
            output is not a column vector
        """

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

        return f

    def get_path_funcs(
        self,
    ) -> tuple[cs.Function | None, cs.Function, cs.Function | None]:
        """
        get all path functions as a tuple (L,f,h)

        Returns
        -------
        tuple[cs.Function|None,cs.Function,cs.Function|None]
            collection of function or None objects
        """
        return (self.L, self.f, self.h)

    def get_event_funcs(self) -> tuple[cs.Function | None, cs.Function]:
        """
        get all evemt functions as a tuple (E,e)

        Returns
        -------
        tuple[cs.Function|None,cs.Function,cs.Function|None]
            collection of function or None objects
        """
        return (self.E, self.e)

    @model_validator(mode="after")
    def validate_trajopt(self) -> "TrajOpt":
        """
        Once basic check on certain fields which hold constants and functions
        are complete, additional checks are done to

        1. check dimension consistency of path and input functions
        2. check shape of dynamics, Lagrange and Mayer terms
        3. validate bounds for variables and constraints

        Returns
        -------
        TrajOpt
            Trajectory optimization object
        """

        self.path_func_ip_check(self.get_path_funcs())
        self.event_func_ip_check(self.get_event_funcs())
        self.func_shapes()
        self.check_lb_ub()

        return self

    def func_shapes(self) -> None:
        """
        check shape of dynamics, Lagrange and Mayer terms

        Returns
        -------
        None

        Raises
        ------
        ValueError
            dynamics vector dont match state dimension
        ValueError
            Lagrange/Mayer are not scalars
        """

        # f is mandatory vector
        if self.f.size_out(0) != (self.nx, 1):
            raise ValueError(
                f"dynamics function ({self.f.name()}) must return a {(self.nx, 1)} vector but received {self.f.size_out(0)}"
            )

        for name, func in zip(["Lagrange", "Mayer"], [self.L, self.E]):
            # L and E are optional (None sets them to zero)
            if func is not None and func.size_out(0) != (1, 1):
                raise ValueError(
                    f"{name} function ({func.name()}) must return a scalar vector but received {func.size_out(0)}"
                )

    def path_func_ip_check(self, funcs) -> None:
        """

        check the size of each input of path function

        Parameters
        ----------
        funcs : _type_
            collection of path functions

        Raises
        ------
        ValueError
            dimension of the path  input variables does not match with dimensions of time, state and control
        """
        inputs = ["t", "x", "u"]
        for func in funcs:
            if func is not None:
                for i, j in enumerate([(1, 1), (self.nx, 1), (self.nu, 1)]):
                    if func.size_in(i) != j:
                        raise ValueError(
                            f"Dimension of {inputs[i]} must be {j} but received {func.size_in(i)} for {func.name()}"
                        )

    def event_func_ip_check(self, funcs) -> None:
        """

        check the size of each input of event function

        Parameters
        ----------
        funcs : _type_
            collection of event functions

        Raises
        ------
        ValueError
            dimension of the event input variables does not match with dimensions of time, state at initial and final points
        """
        inputs = ["t0", "tf", "x0", "xf"]
        for func in funcs:
            if func is not None:
                for i, j in enumerate([(1, 1), (1, 1), (self.nx, 1), (self.nx, 1)]):
                    if func.size_in(i) != j:
                        raise ValueError(
                            f"Dimension of {inputs[i]} must be {j} but received{func.size_in(i)} for {func.name()}"
                        )

    def check_lb_ub(self) -> None:
        """
        1. assign defaults simple bounds for state control and time
        2. checks if bounds are valid

        Raises
        ------
        ValueError
            bounds set for fixed time t0/tf
        ValueError
            final time < initial time
        ValueError
            bounds set for no path constraints
        ValueError
            bounds not set for path constraint
        """

        # process defaults
        # none implies unbounded
        # if not none keep it as it is
        simple = {
            "x_lb": (self.x_lb, (self.nx, 1)),
            "x_ub": (self.x_ub, (self.nx, 1)),
            "u_lb": (self.u_lb, (self.nu, 1)),
            "u_ub": (self.u_ub, (self.nu, 1)),
        }

        for name, obj_and_shape in simple.items():
            if obj_and_shape[0] is None:
                if "_lb" in name:
                    c = -1
                else:
                    c = 1
                # override nones to +/- inf
                setattr(self, name, c * cs.DM.inf() * cs.DM.ones(obj_and_shape[1]))
            # match shapes of x and u, detect nan
            TrajOpt.check_shape(
                name=name, vec=getattr(self, name), known_shape=obj_and_shape[1]
            )
            TrajOpt.check_nan(name=name, vec=getattr(self, name))

        TrajOpt.check_lb_le_ub(self.x_lb, self.x_ub, "x")
        TrajOpt.check_lb_le_ub(self.u_lb, self.u_ub, "u")

        time = {
            "t0_lb": (self.t0_lb, (1, 1)),
            "t0_ub": (self.t0_ub, (1, 1)),
            "tf_lb": (self.tf_lb, (1, 1)),
            "tf_ub": (self.tf_ub, (1, 1)),
        }

        for name, obj_and_shape in time.items():
            # free
            if getattr(self, name[:2]) is None:
                # set default to unassigned time bounds
                if obj_and_shape[0] is None:
                    if "_lb" in name:
                        # positive time
                        setattr(self, name, cs.DM(1e-8))
                    else:
                        # no upper limit
                        setattr(self, name, cs.DM.inf())
                else:
                    # user defined - check quality of data and its shape
                    TrajOpt.check_shape(
                        name=name, vec=obj_and_shape[0], known_shape=obj_and_shape[1]
                    )
                    TrajOpt.check_nan(name=name, vec=obj_and_shape[0])
            else:
                # t0/tf are fixed
                if obj_and_shape[0] is not None:
                    raise ValueError(f"Cannot set bounds to fixed variable {name[:2]}")

        if self.t0 is None:
            TrajOpt.check_lb_le_ub(self.t0_lb, self.t0_ub, "t0")
        if self.tf is None:
            TrajOpt.check_lb_le_ub(self.tf_lb, self.tf_ub, "tf")

        # fixed t0,tf -> tf>t0
        if self.tf is not None and self.t0 is not None and self.t0 > self.tf:
            raise ValueError(
                f"final time must be greater than initial time but t0 ({self.t0})> tf ({self.tf})"
            )

        if self.h is None:
            if self.h_lb is not None or self.h_ub is not None:
                raise ValueError(
                    "Cannot set bounds h_lb or h_ub when path constraint h is not set (None)"
                )
        else:
            for name, bound in zip(["h_lb", "h_ub"], [self.h_lb, self.h_ub]):
                if bound is None:
                    raise ValueError(
                        f"{name} not set for h. Update {name} with valid bounds vector"
                    )
                TrajOpt.check_nan(vec=bound, name=name)
            TrajOpt.check_constraint_shapes(self.h, self.h_lb, self.h_ub, self.h.name())
            TrajOpt.check_lb_le_ub(self.h_lb, self.h_ub, "h")

        TrajOpt.check_nan(vec=self.e_lb, name="e_lb")
        TrajOpt.check_nan(vec=self.e_ub, name="e_ub")
        TrajOpt.check_constraint_shapes(self.e, self.e_lb, self.e_ub, self.e.name())
        TrajOpt.check_lb_le_ub(self.e_lb, self.e_ub, "e")

    @classmethod
    def check_constraint_shapes(cls, g: cs.Function, lb: cs.DM, ub: cs.DM, name: str):
        """
        constraint shape must match with bounds

        Parameters
        ----------
        g : cs.Function
            constraint function
        lb : cs.DM
            lower bound
        ub : cs.DM
            upper bound
        name : str
            function name

        Raises
        ------
        ValueError
            bounds and constraint function dimension dont match
        """
        if not (lb.shape == ub.shape == g.size_out(0)):
            raise ValueError(
                f"The shape of bounds ({name}_lb={lb.shape},{name}_ub={ub.shape}) provided does not match with output of function ({name}={g.size_out(0)})"
            )

    @classmethod
    def check_nan(cls, vec: cs.DM, name: str):
        """
        check of vector contains NaN values

        prevent corruption of data

        Parameters
        ----------
        vec : cs.DM
            constant vector
        name : str
            name

        Raises
        ------
        ValueError
            Nan was there in vec
        """
        if np.isnan(vec.full()).any():
            raise ValueError(f"{name} cannot have NaN")

    @classmethod
    def check_lb_le_ub(cls, lb: cs.DM, ub: cs.DM, name: str) -> None:
        """
        upper bound must be greater than lower bounds vector

        Parameters
        ----------
        lb : cs.DM
            lower bound
        ub : cs.DM
            upper bounds
        name : str
            name of constraint function

        Raises
        ------
        ValueError
            lower bound greater than upper bound
        """
        if np.any(lb.full() > ub.full()):
            raise ValueError(
                f"lower must be <= upper bound for {name} but {name}_lb ({lb}) > {name}_ub ({ub})"
            )

    @classmethod
    def check_shape(cls, vec: cs.DM, known_shape: tuple, name):
        """
        compare shape of matrix with true/expected shape

        Parameters
        ----------
        vec : cs.DM
            constant vector
        known_shape : tuple
            true shape of vector
        name : _type_
            name of the vector used

        Raises
        ------
        ValueError
            shape of vector not matching with the truth
        """
        if vec.shape != known_shape:
            raise ValueError(
                f"{name} is incorrect shape. Expected {known_shape}, got {vec.shape}"
            )

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
            # ZOH to extend control at n+1
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
        if self.method in ["bw_euler", "her_sim", "trap"]:
            print("Costate estimation not implemented for IRK methods")
            trajopt["costate"] = np.full(shape=self.get_costate(nlp.lam_g).shape,fill_value=np.nan)
            trajopt["hamiltonian"] = np.full(shape=(1, n + 1), fill_value=np.nan)
        else:
            if self.L is not None:
                ham_L = self.L.map(n + 1, "serial")(t, X, cs.hcat([U, U[:, -1]]))
            else:
                ham_L = 0
            trajopt["costate"] = self.get_costate(nlp.lam_g)
            trajopt["hamiltonian"] = ham_L + cs.sum1(
                trajopt["costate"]
                * self.f.map(n + 1, "serial")(t, X, cs.hcat([U, U[:, -1]]))
            )

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
        elif self.method == "exp_mid":
            Fm = F(0.5 * (tl + tr), Xl + h / 2 * F(tl, Xl, U), U)
            defect = dX - h * Fm
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
        elif self.method == "casadi_integrator":
            phi = cs.integrator(
                "phi",
                self.method.plugin,
                self.method._scaled_ode,
                0,
                1,
                self.method.plugin_options,
            )
            Phi = phi.map(self.n, "serial")
            defect = Xr - Phi(x0=Xl, p=cs.vcat([U, tl, tr]))["xf"]
            return defect
        else:
            raise ValueError("Wrong option for method")

    def get_costate(self, lam_g):
        nx = self.nx
        n = self.n
        ne = self.e.size_out(0)[0]
        lam_x0 = -lam_g[0:nx, :]
        lam_dyn = -lam_g[ne : ne + nx * n, :]
        costate = cs.hcat([lam_x0, cs.reshape(lam_dyn, nx, n)])
        return costate


def plotting(t: np.ndarray, var: np.ndarray, name: str, pdf: PdfPages):
    plt.style.use('pub_qual.mplstyle')
    for i in range(var.shape[0]):
        plt.figure()

        if name == "u":
            plt.step(t.ravel()[0:-1], var[i, :].ravel())
        else:
            if np.isnan(var).all():
                print(
                    "Costate estimation not implemented for bw_euler, trap and her_sim. User fw_euler or exp_mid"
                )
            plt.plot(t.ravel(), var[i, :].ravel())
        plt.ylabel(rf"${name}_{i}$")
        plt.xlabel("time [s]")
        plt.grid(True)
        pdf.savefig(bbox_inches="tight")
        plt.close()


def plot_solution(sol: cs.OptiSol, nlp: dict):
    txlam = sol.value(nlp["t"])
    tu = txlam
    X = sol.value(nlp["X"]).reshape(nlp["X"].shape)
    U = sol.value(nlp["U"]).reshape(nlp["U"].shape)
    lam = sol.value(nlp["costate"]).reshape(nlp["costate"].shape)
    ham = sol.value(nlp["hamiltonian"]).reshape(nlp["hamiltonian"].shape)

    with PdfPages("solution.pdf") as pdf:
        for name, var in zip(["x", "u", r"\lambda", r"H"], [X, U, lam, ham]):
            t = txlam if name != "u" else tu
            plotting(t, var, name, pdf=pdf)


