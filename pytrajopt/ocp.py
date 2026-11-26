import casadi as cs
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    field_validator,
    model_validator,
)


class Ocp(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    # dimensions of state control and discretization intervals
    nx: PositiveInt
    nu: PositiveInt
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

    @field_validator("t0", "tf")
    @classmethod
    def positive(cls, t: cs.DM, data: ValidationInfo) -> cs.DM:
        if t is not None and t < 0:
            raise ValueError(f"{data.field_name} must be positive but received {t} <0 ")
        return t

    @model_validator(mode="before")
    @classmethod
    def type_cast(cls, raw) -> dict:
        allowed = [
            "t0",
            "tf",
            "e_lb",
            "e_ub",
            "h_lb",
            "h_ub",
            "x_lb",
            "x_ub",
            "u_lb",
            "u_ub",
            "t0_lb",
            "t0_ub",
            "tf_lb",
            "tf_ub",
            "J_s",
            "x_s",
            "x_o",
            "u_s",
            "u_o",
            "t_s",
            "t_o",
            "e_s",
            "h_s",
        ]
        for name, value in raw.items():
            if name in allowed:
                raw[name] = Ocp.py2cs(value, name)
        return raw

    @classmethod
    def py2cs(cls, var, name):
        if isinstance(var, (int, float, list, tuple)):
            if isinstance(var, (float, int)):
                return cs.DM(var)
            else:
                if not all(isinstance(i, (float, int, cs.DM)) for i in var):
                    raise TypeError(
                        f"All elements of {name} must be a float or int but received {type(var)}"
                    )
                return cs.DM(var)
        elif isinstance(var, (cs.DM)) or var is None:
            return var
        else:
            raise TypeError(f"{name} expects list, tuple, DM")

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
            names = ["t_s", "t_o"]
            defaults = [cs.DM(1), cs.DM(0)]
            for name, default in zip(names, defaults):
                setattr(self, name, default)
                Ocp.check_nan(getattr(self, name), name)
                Ocp.check_shape(getattr(self, name), (1, 1), name)

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


class DisOcp(BaseModel):
    """
    Class which has all expressions related to the discrete OCP

    Post solving the NLP via opti, they can be accessed and numerically
    evaluate via Optisol

    Shapes checked post creation to ensure transcription was done right
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    t: cs.MX | cs.DM
    t0: cs.MX | cs.DM | float | int
    tf: cs.MX | cs.DM | float | int
    X: cs.MX
    U: cs.MX
    L: cs.MX | cs.DM | float | int
    E: cs.MX | cs.DM | float | int
    J: cs.MX | cs.DM | float | int
    f: cs.MX
    e: cs.MX
    h: cs.MX | None
    dual: dict[str, cs.MX | np.ndarray | None]
    nx: int
    nu: int
    n: int
    ne: int
    nh: int | None

    @model_validator(mode="after")
    def check(self) -> "DisOcp":
        """
        Checks the shape of every expression associated with discrete OCP
        obtained from transcription.

        Acts as check on transcription itself and help to find bugs

        Returns
        -------
        DisOcp
            A class which holds the discrete OCP
        """

        self.check_shape(self.t, (1, self.n + 1), "time vector")
        self.check_shape(self.t0, (1, 1), "t0")
        self.check_shape(self.tf, (1, 1), "tf")

        self.check_shape(self.X, (self.nx, self.n + 1), "State matrix")
        self.check_shape(self.U, (self.nu, self.n), "Control matrix")

        self.check_shape(self.L, (1, 1), "Lagrange objective")
        self.check_shape(self.E, (1, 1), "Mayer objective")
        self.check_shape(self.J, (1, 1), "Total objective")
        self.check_shape(
            self.f, (self.nx, self.n + 1), "dynamics matrix (f of xd=f(x,u))"
        )
        self.check_shape(self.e, (self.ne, 1), "event constraint")
        self.check_shape(self.h, (self.nh, self.n + 1), "path constraint")

        shapes = {
            "e": (self.ne, 1),
            "f": (self.nx, self.n),
            "X": (self.nx, self.n + 1),
            "U": (self.nu, self.n),
            "t0": (1, 1),
            "tf": (1, 1),
            "dt": (1, self.n + 1),
        }
        if self.nh is not None:
            shapes.update({"h": (self.nh, self.n + 1)})
        for name, dual in self.dual.items():
            if "kkt" in name:
                self.check_shape(
                    dual,
                    shapes[name.rsplit("_", 1)[-1]],
                    f"KKT mutipliers of {name.rsplit('_', 1)[-1]} constraint",
                )
            if "lam" in name and any(i in name for i in ("x", "e")):
                self.check_shape(dual, shapes[name[-1]], f"Costate estimate of {name}")
        return self

    @classmethod
    def check_shape(cls, exp: cs.MX | cs.DM | float, shape: tuple, name: str):
        """
        Checks if predefined shapes match

        If the shape has none, the OCP may not have given expression

        If expression is none, KKT/costate estimates were not defined

        Parameters
        ----------
        exp : cs.MX | cs.DM | float
            expression
        shape : tuple
            given shape
        name : str
            name of expression

        Raises
        ------
        ValueError
            Shape mismatch
        """
        if None in shape:
            print(f"{name} not in OCP, shape checks ignored")
            return
        if exp is None:
            print(f"{name} not implemented")
            return

        if isinstance(exp, (cs.DM, cs.MX)) and exp.shape != shape:
            raise ValueError(
                f"Discrete {name} must be of shape {shape} but received {exp.shape}"
            )
