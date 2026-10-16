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
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline


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
    def check_length(cls, path: list[cs.SX]) -> list[cs.SX]:
        """

        length of path list must be 3 [t,x,u]

        Parameters
        ----------
        path : list[cs.SX]
            the path variables

        Returns
        -------
        list[cs.SX]
            validated path variables

        Raises
        ------
        ValueError
            length of the list is not 3
        """
        if len(path) != 3:
            raise ValueError(
                f"t,x,u must be present in path. Recived {len(path)} expected 3"
            )
        return path

    @field_validator("f")
    @classmethod
    def check_column(cls, f: cs.SX) -> cs.SX:
        """

        the dynamics vector must be a column vector

        Parameters
        ----------
        f : cs.SX
            xd=f(t,x,u)

        Returns
        -------
        cs.SX
            validated f vector

        Raises
        ------
        ValueError
            if a tow vector or matrix is passed
        """
        if f.shape[1] != 1:
            raise ValueError(
                f"f must be a column vector but received f of shape {f.shape}"
            )
        return f

    def __eq__(self, other: str) -> bool:
        """
        casadi_integrator=='casadi_integrator' must return true

        this functionality added to keep Ocp code clean

        Parameters
        ----------
        other : str
            string to be compared against

        Returns
        -------
        bool
            truth value
        """
        # If the other object is a string, check if it matches the class name
        if isinstance(other, str):
            return self.__class__.__name__ == other

        # Fallback to standard comparison for other object types
        return super().__eq__(other)

    @classmethod
    def get_scaled_dynamics(cls, path: list, f: cs.SX) -> dict[str, cs.SX]:
        """
        transform time from [t0,tf]->[0,1] (tau)

        t0 tf are parameters in the ode (constants along the control vector)

        helps to reuse casadi integrator object

        Parameters
        ----------
        path : []
            path variables [t,x,u]
        f : cs.SX
            ode expression [xd=f(t,x,u)]

        Returns
        -------
        dict[str,cs.SX]
            transformed ode in a dict
        """

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
    def generate_ode_dict(self) -> "casadi_integrator":
        """
        casadi.integrator compatible dict generation

        Returns
        -------
        casadi_integrator
            all objects needed to create a casadi integrator

        Raises
        ------
        ValueError
            shape mismatch of path variables
        """
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


class Ocp(BaseModel):
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
    # scale
    J_s: cs.DM | None = None
    x_s: cs.DM | None = None
    u_s: cs.DM | None = None
    e_s: cs.DM | None = None
    h_s: cs.DM | None = None
    t_s: cs.DM | None = None
    # offset
    x_o: cs.DM | None = None
    u_o: cs.DM | None = None
    t_o: cs.DM | None = None
    # obj tuple[cs.DM,cs.DM] | None (1,0)
    # constriant
    # variables

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
    def validate_Ocp(self) -> "Ocp":
        """
        Once basic check on certain fields which hold constants and functions
        are complete, additional checks are done to

        1. check dimension consistency of path and input functions
        2. check shape of dynamics, Lagrange and Mayer terms
        3. validate bounds for variables and constraints

        Returns
        -------
        Ocp
            Trajectory optimization object
        """

        self.path_func_ip_check(self.get_path_funcs())
        self.event_func_ip_check(self.get_event_funcs())
        self.func_shapes()
        self.check_lb_ub()
        self.check_scale_offset()

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
            Ocp.check_shape(
                name=name, vec=getattr(self, name), known_shape=obj_and_shape[1]
            )
            Ocp.check_nan(name=name, vec=getattr(self, name))

        Ocp.check_lb_le_ub(self.x_lb, self.x_ub, "x")
        Ocp.check_lb_le_ub(self.u_lb, self.u_ub, "u")

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
                    Ocp.check_shape(
                        name=name, vec=obj_and_shape[0], known_shape=obj_and_shape[1]
                    )
                    Ocp.check_nan(name=name, vec=obj_and_shape[0])
            else:
                # t0/tf are fixed
                if obj_and_shape[0] is not None:
                    raise ValueError(f"Cannot set bounds to fixed variable {name[:2]}")

        if self.t0 is None:
            Ocp.check_lb_le_ub(self.t0_lb, self.t0_ub, "t0")
        if self.tf is None:
            Ocp.check_lb_le_ub(self.tf_lb, self.tf_ub, "tf")

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
                Ocp.check_nan(vec=bound, name=name)
            Ocp.check_constraint_shapes(self.h, self.h_lb, self.h_ub, self.h.name())
            Ocp.check_lb_le_ub(self.h_lb, self.h_ub, "h")

        Ocp.check_nan(vec=self.e_lb, name="e_lb")
        Ocp.check_nan(vec=self.e_ub, name="e_ub")
        Ocp.check_constraint_shapes(self.e, self.e_lb, self.e_ub, self.e.name())
        Ocp.check_lb_le_ub(self.e_lb, self.e_ub, "e")

    def check_scale_offset(self):

        # assign defaults
        # same as calling them without scaling them

        names = ["J_s", "x_s", "u_s", "e_s"]
        shapes = [(1, 1), (self.nx, 1), (self.nu, 1), self.e.size_out(0)]

        for name, shape in zip(names, shapes):
            if getattr(self, name) is None:
                setattr(self, name, cs.DM.ones(shape))
            Ocp.check_nan(getattr(self, name), name)
            Ocp.check_shape(getattr(self, name), shape, name)

        names = ["x_o", "u_o"]
        shapes = [(self.nx, 1), (self.nu, 1)]

        for name, shape in zip(names, shapes):
            if getattr(self, name) is None:
                setattr(self, name, cs.DM.zeros(shape))
            Ocp.check_nan(getattr(self, name), name)
            Ocp.check_shape(getattr(self, name), shape, name)

        if self.J_s is not None and self.J_s <= 0:
            raise ValueError(
                "Objective scaling factor must be positive. Negative scaling not allowed"
            )

        if self.t0 is None or self.tf is None:
            names = ["t_s", "t_o"]
            defaults = [cs.DM(1), cs.DM(0)]
            for name, default in zip(names, defaults):
                if getattr(self, name) is None:
                    setattr(self, name, default)
                Ocp.check_nan(getattr(self, name), name)
                Ocp.check_shape(getattr(self, name), (1, 1), name)
        else:
            if self.t_s is not None or self.t_o is not None:
                raise ValueError(
                    "Cannot set scales for time variables when both t0 and tf are fixed"
                )

        if self.h is not None:
            if self.h_s is None:
                self.h_s = cs.DM.ones(self.h.size_out(0))
            Ocp.check_nan(self.h_s, "h_s")
            Ocp.check_shape(self.h_s, self.h.size_out(0), "h_s")
        else:
            if self.h_s is not None:
                raise ValueError("Cannot set scales for OCP with no path constraints")

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

def generate_NLP(ocp:Ocp,nlp:cs.Opti):
    n = ocp.n
    nx, nu = ocp.nx, ocp.nu
    dec_var = []
    if ocp.t0 is None:
        t0 = nlp.variable()
        dec_var.append(t0)
        nlp.set_linear_scale(t0, ocp.t_s, ocp.t_o)
    else:
        t0 = ocp.t0
    if ocp.tf is None:
        tf = nlp.variable()
        dec_var.append(tf)
        nlp.set_linear_scale(tf, ocp.t_s, ocp.t_o)
    else:
        tf = ocp.tf
    t = cs.linspace(t0, tf, n + 1).T
    X = nlp.variable(nx, n + 1)
    nlp.set_linear_scale(
        X, cs.repmat(ocp.x_s, 1, ocp.n + 1), cs.repmat(ocp.x_o, 1, ocp.n + 1)
    )
    dec_var.append(X)
    U = nlp.variable(nu, n)
    nlp.set_linear_scale(
        U, cs.repmat(ocp.u_s, 1, ocp.n), cs.repmat(ocp.u_o, 1, ocp.n)
    )
    dec_var.append(U)
    # decision vector
    W = cs.vcat([cs.vec(i) for i in dec_var])
    event_ip = (t[:, 0], t[:, -1], X[:, 0], X[:, -1])
    # objective
    if ocp.L is None:
        L = 0
    else:
        L = (
            (tf - t0)
            / n
            * cs.sum2(ocp.L.map(n, "serial")(t[:, 0:-1], X[:, 0:-1], U))
        )
    if ocp.E is None:
        E = 0
    else:
        E = ocp.E(*event_ip)
    objective = L + E
    cons = []
    cons_lb = []
    cons_ub = []
    cons_scale = []
    # event constraints
    e = ocp.e(*event_ip)
    cons.append(e)
    cons_lb.append(ocp.e_lb)
    cons_ub.append(ocp.e_ub)
    cons_scale.append(ocp.e_s)
    # dynamics
    f = get_discrete_dynamics(ocp,t, X, U)
    cons.append(f)
    zr = cs.DM.zeros(f.shape)
    cons_lb.append(zr)
    cons_ub.append(zr)
    cons_scale.append(cs.repmat(ocp.x_s, 1, n))
    # path constriants
    if ocp.h is not None:
        # ZOH to extend control at n+1
        h = ocp.h.map(n + 1, "serial")(t, X, cs.hcat([U, U[:, -1]]))
        cons.append(h)
        cons_lb.append(cs.repmat(ocp.h_lb, 1, n + 1))
        cons_ub.append(cs.repmat(ocp.h_ub, 1, n + 1))
        cons_scale.append(cs.repmat(ocp.h_s, 1, n + 1))
    # constraints
    G = cs.vcat([cs.vec(i) for i in cons])
    G_lb = cs.vcat([cs.vec(i) for i in cons_lb])
    G_ub = cs.vcat([cs.vec(i) for i in cons_ub])
    G_s = cs.vcat([cs.vec(i) for i in cons_scale])
    # simple bounds
    cons_simple_lb = []
    cons_simple_ub = []
    cons_simple_scale = []
    if ocp.t0 is None:
        t0_lb = ocp.t0_lb
        t0_ub = ocp.t0_ub
        cons_simple_lb.append(t0_lb)
        cons_simple_ub.append(t0_ub)
        cons_simple_scale.append(ocp.t_s)
    if ocp.tf is None:
        tf_lb = ocp.tf_lb
        tf_ub = ocp.tf_ub
        cons_simple_lb.append(tf_lb)
        cons_simple_ub.append(tf_ub)
        cons_simple_scale.append(ocp.t_s)
    X_lb = cs.repmat(ocp.x_lb, 1, n + 1)
    X_ub = cs.repmat(ocp.x_ub, 1, n + 1)
    cons_simple_lb.append(X_lb)
    cons_simple_ub.append(X_ub)
    cons_simple_scale.append(cs.repmat(ocp.x_s, 1, n + 1))
    U_lb = cs.repmat(ocp.u_lb, 1, n)
    U_ub = cs.repmat(ocp.u_ub, 1, n)
    cons_simple_lb.append(U_lb)
    cons_simple_ub.append(U_ub)
    cons_simple_scale.append(cs.repmat(ocp.u_s, 1, n))
    W_lb = cs.vcat([cs.vec(i) for i in cons_simple_lb])
    W_ub = cs.vcat([cs.vec(i) for i in cons_simple_ub])
    W_s = cs.vcat([cs.vec(i) for i in cons_simple_scale])
    # assemble
    nlp.minimize(objective, float(ocp.J_s))
    nlp.subject_to(nlp.bounded(G_lb, G, G_ub), G_s)
    nlp.subject_to(nlp.bounded(W_lb, W, W_ub), W_s)
    dis_ocp = {}
    dis_ocp["t"] = t
    dis_ocp["t0"] = t0
    dis_ocp["tf"] = tf
    dis_ocp["X"] = X
    dis_ocp["U"] = U
    dis_ocp["dynamics"] = f
    dis_ocp["objective"] = objective
    if ocp.method in ["bw_euler", "her_sim", "trap"]:
        print("Costate estimation not implemented for IRK methods")
        dis_ocp["costate"] = np.full(
            shape=get_costate(ocp,nlp.lam_g).shape, fill_value=np.nan
        )
    else:
        dis_ocp["costate"] = get_costate(ocp,nlp.lam_g)
    return dis_ocp

def get_discrete_dynamics(ocp:Ocp, t, X, U) -> cs.MX:
    F = ocp.f.map(ocp.n, "serial")
    Xl = X[:, 0:-1]
    Xr = X[:, 1:]
    tl = t[:, 0:-1]
    tr = t[:, 1:]
    h = t[:, 1] - t[:, 0]
    dX = Xr - Xl
    Fl = F(tl, Xl, U)
    Fr = F(tr, Xr, U)
    if ocp.method == "bw_euler":
        defect = dX - Fr * h
        return defect
    elif ocp.method == "fw_euler":
        defect = dX - Fl * h
        return defect
    elif ocp.method == "exp_mid":
        Fm = F(0.5 * (tl + tr), Xl + h / 2 * F(tl, Xl, U), U)
        defect = dX - h * Fm
        return defect
    elif ocp.method == "trap":
        defect = dX - 0.5 * h * (Fl + Fr)
        return defect
    elif ocp.method == "her_sim":
        Xm = 0.5 * (Xr + Xl) + h / 8 * (Fl - Fr)
        tm = 0.5 * (tr + tl)
        Fm = F(tm, Xm, U)
        defect = dX - h / 6 * (Fl + 4 * Fm + Fr)
        return defect
    elif ocp.method == "casadi_integrator":
        phi = cs.integrator(
            "phi",
            ocp.method.plugin,
            ocp.method._scaled_ode,
            0,
            1,
            ocp.method.plugin_options,
        )
        Phi = phi.map(ocp.n, "serial")
        defect = Xr - Phi(x0=Xl, p=cs.vcat([U, tl, tr]))["xf"]
        return defect
    else:
        raise ValueError("Wrong option for method")

def get_costate(ocp:Ocp, lam_g):
    nx = ocp.nx
    n = ocp.n
    ne = ocp.e.size_out(0)[0]
    lam_dyn = -lam_g[ne : ne + nx * n, :]
    costate = cs.reshape(lam_dyn, nx, n)
    return costate

def plotting(t: np.ndarray, var: np.ndarray, name: str, pdf: PdfPages):
    # plt.style.use("./pub_qual.mplstyle")
    for i in range(var.shape[0]):
        plt.figure()

        if name == "u":
            plt.step(t.ravel(), var[i, :].ravel())
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


def plot_solution(sol: cs.OptiSol, dis_ocp: dict,name:str='solution'):
    tx = sol.value(dis_ocp["t"])
    tu = tx[0:-1]
    tlam = tx[1:]
    tvars = [tx, tu, tlam]
    X = sol.value(dis_ocp["X"]).reshape(dis_ocp["X"].shape)
    U = sol.value(dis_ocp["U"]).reshape(dis_ocp["U"].shape)
    lam = sol.value(dis_ocp["costate"]).reshape(dis_ocp["costate"].shape)

    with PdfPages(f"{name}.pdf") as pdf:
        for name, var, tvar in zip(["x", "u", r"\lambda"], [X, U, lam], tvars):
            plotting(tvar, var, name, pdf=pdf)


def pc_interp(t: np.ndarray, U: np.ndarray):
    if U.ndim == 2:
        U = np.hstack([U, U[:, -1].reshape(-1,1)])
        axis = 1
    else:
        U = np.hstack([U, U[-1]])
        axis = 0
    zoh = make_interp_spline(t, U, k=0, axis=axis)
    return zoh


def check_feasibility(ocp:Ocp,dis_ocp:dict,sol:cs.OptiSol,name:str='feasibility'):

    X=sol.value(dis_ocp['X'])
    U=sol.value(dis_ocp['U'])
    tg=sol.value(dis_ocp['t'])
    x0=X[:,0]
    f=ocp.f


    zoh = pc_interp(tg, U)

    def f_solve_ivp(t, x):
        return f(t, x, zoh(t)).full().flatten()

    td = np.linspace(tg[0], tg[-1], len(tg) * 10)
    options = {"rtol": 1e-12, "atol": 1e-12}
    ivp_sol = solve_ivp(f_solve_ivp, [tg[0],tg[-1]], x0, "LSODA", td, options)
    if ivp_sol.success is False:
        raise ValueError("Time marching failed")

    with PdfPages(f"{name}.pdf") as pdf:
        for i in range(x0.size):
            plt.figure()
            if X.ndim==1:
                plt.plot(tg,X)
                plt.plot(td,ivp_sol.y,label='LSODA')
            else:
                plt.plot(tg,X[i,:],'o',label='NLP')
                plt.plot(td,ivp_sol.y[i,:],'--',label='LSODA')
            plt.xlabel('time [s]')
            plt.ylabel(rf'$x_{i}$')
            plt.grid(True)
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()

        if U.ndim==1:
            nu=1
        else:
            nu=U.shape[0]
        Ud=zoh(td)
        for i in range(nu):
            if U.ndim==1:
                plt.plot(tg[0:-1],U,'o',label='NLP')
                plt.plot(td,Ud,'--',label='ZOH')
            else:
                plt.plot(tg[0:-1],U[i,:],'o',label='dis_ocp')
                plt.plot(td,Ud[i,:],'--',label='ZOH')
            plt.xlabel('time [s]')
            plt.ylabel(rf'$u_{i}$')
            plt.grid(True)
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()

def plot_solution_all_phase(dis_ocps:list,sol:cs.OptiSol):

    with PdfPages('all.pdf') as pdf:
        for j in range(3):
            plt.figure()
            for i, dis_ocp in enumerate(dis_ocps):
                t=sol.value(dis_ocp["t"]).ravel()
                X=sol.value(dis_ocp["X"])
                if X.ndim==1:
                    plt.plot(t,X,label=f'phase {i}')
                else:
                    plt.plot(t,X[j,:].ravel(),label=f'phase {i}')
            plt.xlabel('time [s]')
            plt.ylabel(f'$x_{j}$')
            plt.grid()
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()

        for j in range(1):
            plt.figure()
            for i, dis_ocp in enumerate(dis_ocps):
                t=sol.value(dis_ocp["t"])
                U=sol.value(dis_ocp["U"])
                zoh=pc_interp(t,U)
                zoh=zoh(t)
                if zoh.ndim==1:
                    plt.plot(t,zoh,label=f'phase {i}')
                else:
                    plt.plot(t,zoh[j,:].ravel(),label=f'phase {i}')
            plt.xlabel('time [s]')
            plt.ylabel(f'$u_{j}$')
            plt.grid()
            plt.legend()
            pdf.savefig(bbox_inches="tight")
            plt.close()