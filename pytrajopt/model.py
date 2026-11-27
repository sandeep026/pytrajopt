from typing import Literal

import casadi as cs
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)


class Expression(BaseModel):
    """
    A class which holds the expressions
    related to the optimal control problem
    like the lagrange and mayer objective
    (L,E), dynamics (xd=f), event constraints
    (<=e(t0,tf,x0,xf)<=) and path (<=h(t,x,u)<=)
    constraints.

    This class checks type and if the objective is
    a scalar.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    L: cs.SX | None = None
    E: cs.SX | None = None
    f: cs.SX | None = None
    e: cs.SX | None = None
    h: cs.SX | None = None

    @field_validator("L", "E")
    @classmethod
    def check_scalar(cls, obj: cs.SX | None, info: ValidationInfo) -> cs.SX | None:
        """
        Objective function related terms must be scalar.
        """
        if obj is not None and obj.numel() != 1:
            raise ValueError(
                f"{info.field_name} must be scalar but received matrix of shape {obj.shape}"
            )
        return obj


class Syms(BaseModel):
    """
    Symbolic variables for the defining the optimal
    control problem problem.

    Generate symbolics for the OCP and convert the
    expressions to functions.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    nx: PositiveInt
    nu: PositiveInt
    _event: list[cs.SX] = PrivateAttr(default_factory=list)
    _path: list[cs.SX] = PrivateAttr(default_factory=list)
    _exps: Expression | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def create_syms(self) -> "Syms":
        """
        generates symbolics for
        t0,tf,x0,xf and t,x,u
        which will be used to create
        casadi functions

        Returns
        -------
        Syms
            Syms
        """
        ts = cs.SX.sym("ts", 1, 1)
        xs = cs.SX.sym("xs", self.nx, 1)
        us = cs.SX.sym("us", self.nu, 1)
        t0s = cs.SX.sym("t0s", 1, 1)
        tfs = cs.SX.sym("tfs", 1, 1)
        x0s = cs.SX.sym("x0s", self.nx, 1)
        xfs = cs.SX.sym("xfs", self.nx, 1)
        self._path = [ts, xs, us]
        self._event = [t0s, tfs, x0s, xfs]
        return self

    def get_path(self) -> list[cs.SX]:
        """
        t,x,u

        Returns
        -------
        list[cs.SX]
            [t,x,u]
        """
        return self._path

    def get_event(self) -> list[cs.SX]:
        """
        t0,tf,x0,xf

        Returns
        -------
        list[cs.SX]
            [t0,tf,x0,xf]
        """
        return self._event

    def get_function(
        self, name: str, expression: cs.SX, nature: Literal["event", "path"]
    ) -> cs.Function:
        """
        generates a casadi function for Lagrange, Mayer,
        event, dynamics and path constraints

        Parameters
        ----------
        name : str
            name passed to casadi.Function
        expression : cs.SX
            expression of symbolics
        nature : Literal[&#39;event&#39;,&#39;path&#39;]
            path or event variables

        Returns
        -------
        cs.Function
            a casadi function of the expression

        Raises
        ------
        ValueError
            path constraints have event variables and vice versa
        """
        input_vars = self.get_path() if nature == "path" else self.get_event()
        try:
            return cs.Function(name, input_vars, [expression])
        except RuntimeError:
            raise ValueError(
                f"Cannot mix path (t,x,u) and event (t0,tf,x0,xf) variables in {name}"
            )

    def collect_expression(self, data) -> None:
        """
        store the expressions provided by the user
        """
        if self._exps is not None:
            print("Overwriting previous expressions")
        self._exps = Expression(**data)

    def generate_funcs(self) -> dict[str, cs.Function]:
        """
        create a dict of functions based the OCP
        expressions created
        """
        path = ["L", "f", "h"]
        event = ["E", "e"]
        func_dict = {}

        for name, nature in zip(path, ["path"] * 3):
            if getattr(self._exps, name) is not None:
                func_dict[name] = self.get_function(
                    name, getattr(self._exps, name), nature
                )

        for name, nature in zip(event, ["event"] * 2):
            if getattr(self._exps, name) is not None:
                func_dict[name] = self.get_function(
                    name, getattr(self._exps, name), nature
                )

        return func_dict
