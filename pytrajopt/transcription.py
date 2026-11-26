from typing import Literal

import casadi as cs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)


class DynamicMesh(BaseModel):
    scale: float = 1  # >0
    offset: float = 0
    k1: float = 0.1  # >0
    k2: float = 2  # >1


class Integrator(BaseModel):
    # provides access to casadi integrators
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    plugin: Literal[
        "cvodes",
        "collocation",
        "rk",
        "fw_euler",
        "bw_euler",
        "exp_mid",
        "trap",
        "her_sim",
        "rk4",
    ] = "rk4"
    plugin_options: dict = Field(default_factory=dict)
    nx: int
    nu: int
    path: list[cs.SX]
    f: cs.SX
    _scaled_ode: dict[str, cs.SX] = PrivateAttr(default_factory=dict)
    _fs: cs.Function = PrivateAttr(default_factory=cs.Function)

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
    def get_scaled_dynamics(
        cls, path: list, f: cs.SX
    ) -> tuple[dict[str, cs.SX], cs.Function]:
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
        fs = cs.Function("fs", [ode_s["t"], ode_s["x"], ode_s["p"]], [ode_s["ode"]])
        return ode_s, fs

    @model_validator(mode="after")
    def generate_ode_dict(self) -> "Integrator":
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

        self._scaled_ode, self._fs = self.get_scaled_dynamics(self.path, self.f)

        custom_methods = ["fw_euler", "bw_euler", "exp_mid", "trap", "her_sim", "rk4"]
        if self.plugin in custom_methods:
            if self.plugin_options:
                raise ValueError(f"plugin_options cannot be set for {custom_methods}")
        return self


# fw_euler - forward Euler
# bw_euler - backward Euler
# rk4      - 4th order rk
# trap     - trapezoidal
# her_sim  - Hermite Simpson


class Transcription(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    n: PositiveInt
    integrator: Integrator
    grid: cs.DM | None = None

    @model_validator(mode="after")
    def check_grid(self) -> "Transcription":
        if self.grid is None:
            self.grid = cs.linspace(0, 1, self.n + 1).T
        if self.grid.shape != (1, self.n + 1):
            raise ValueError(
                f"Grid shape must be {(1, self.n + 1)} but received {self.grid.shape}"
            )
        if (cs.diff(self.grid) < 0).is_one():
            raise ValueError("Normalized grid width must be positive")
        if self.grid[0] != 0 or self.grid[-1] != 1:
            raise ValueError(
                f"Normalized grid must start and end at 0 and  but recieved grid[0]={self.grid[0]} and grid[-1]={self.grid[-1]}"
            )
        return self

    def get_defect(self) -> cs.Function:
        fs = self.integrator._fs
        tl = cs.SX.sym("tl", 1, 1)
        tr = cs.SX.sym("tr", 1, 1)
        xl = cs.SX.sym("xl", self.integrator.nx, 1)
        xr = cs.SX.sym("xr", self.integrator.nx, 1)
        u = cs.SX.sym("u", self.integrator.nu, 1)
        p = cs.vertcat(u, tl, tr)

        if self.integrator.plugin == "rk4":
            k1 = fs(0, xl, p)
            k2 = fs(0.5, xl + 0.5 * k1, p)
            k3 = fs(0.5, xl + 0.5 * k2, p)
            k4 = fs(1, xl + k3, p)
            defect = xr - xl - 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif self.integrator.plugin == "fw_euler":
            kl = fs(0, xl, p)
            defect = xr - xl - (kl)
        elif self.integrator.plugin == "bw_euler":
            kr = fs(1, xr, p)
            defect = xr - xl - (kr)
        elif self.integrator.plugin == "trap":
            kl = fs(0, xl, p)
            kr = fs(1, xr, p)
            defect = xr - xl - 0.5 * (kl + kr)
        elif self.integrator.plugin == "exp_mid":
            xm = xl + 0.5 * fs(0, xl, p)
            km = fs(0.5, xm, p)
            defect = xr - xl - km
        elif self.integrator.plugin == "her_sim":
            kl = fs(0, xl, p)
            kr = fs(1, xr, p)
            xm = 0.5 * (xr + xl) + 1 / 8 * (kl - kr)
            tm = 0.5 * (tr + tl)
            km = fs(tm, xm, p)
            defect = xr - xl - 1 / 6 * (kl + 4 * km + kr)
        elif self.integrator.plugin in ["rk", "cvodes", "collocation"]:
            phi = cs.integrator(
                "phi",
                self.integrator.plugin,
                self.integrator._scaled_ode,
                0,
                [1],
                self.integrator.plugin_options,
            )
            defect = xr - phi(x0=xl, p=p)["xf"]
        else:
            raise ValueError("Incorrect numerical method for Intergator")
        defect = cs.Function("defect", [tl, tr, xl, xr, u], [defect])
        return defect
