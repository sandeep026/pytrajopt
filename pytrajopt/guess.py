import casadi as cs
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    model_validator,
)


class LinGuess(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    n: PositiveInt
    xl: list[float | int]
    xr: list[float | int]
    ul: list[float | int]
    ur: list[float | int]
    tl: float | int = Field(ge=0, default=0)
    tr: float | int = Field(gt=0, default=1)
    _Xg: cs.DM | None = PrivateAttr(default=None)
    _Ug: cs.DM | None = PrivateAttr(default=None)
    _tg: cs.DM | None = PrivateAttr(default=None)
    _t0g: cs.DM | None = PrivateAttr(default=None)
    _tfg: cs.DM | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def check(self) -> "LinGuess":

        for name, limits in zip(
            ["state", "control"], [(self.xl, self.xr), (self.ul, self.ur)]
        ):
            if len(limits[0]) != len(limits[1]):
                raise ValueError(
                    f"Number of elements mismatch for {name} end points. Received l={len(limits[0])} r={len(limits[1])}"
                )

        if self.tl < 0 or self.tr <= 0:
            raise ValueError(
                f"Guess for time cannot be negative (t0={self.tl},tf={self.tr}) "
            )
        if self.tr - self.tl <= 0:
            raise ValueError(
                f"Final time guess must be greater than initial time but received tr={self.tr}<= tl={self.tl}"
            )

        self._Xg = cs.vcat(
            [cs.linspace(l, r, self.n + 1).T for l, r in zip(self.xl, self.xr)]
        )
        self._Ug = cs.vcat(
            [cs.linspace(l, r, self.n).T for l, r in zip(self.ul, self.ur)]
        )
        self._t0g = cs.DM(self.tl)
        self._tfg = cs.DM(self.tr)
        self._tg = cs.linspace(self.tl, self.tr, self.n + 1).T
        return self

    def set_guess(self, dis_ocp: DisOcp, nlp: cs.Opti) -> None:
        """
        set guess to NLP

        Parameters
        ----------
        dis_ocp : DisOcp
            discrete OCP
        nlp : cs.Opti
            Opti
        """

        for name, att in zip(["state", "control"], ["X", "U"]):
            if getattr(dis_ocp, att).shape != getattr(self, f"_{att}g").shape:
                raise ValueError(
                    f"Guess for the {name} matrix has incorrect dimension, expected {getattr(dis_ocp, att).shape} but received {getattr(self, f'_{att}g').shape}"
                )

        nlp.set_initial(dis_ocp.X, self._Xg)
        nlp.set_initial(dis_ocp.U, self._Ug)

        if not isinstance(dis_ocp.t, cs.DM) or not dis_ocp.t.is_constant():
            for i in cs.symvar(dis_ocp.t):
                if i.shape == (1, self.n + 1):
                    nlp.set_initial(dis_ocp.t, self._tg)
                    break
                if i.shape == (1, 1):
                    if cs.is_equal(dis_ocp.t0, i):
                        nlp.set_initial(dis_ocp.t0, self._t0g)
                    elif cs.is_equal(dis_ocp.tf, i):
                        nlp.set_initial(dis_ocp.tf, self._tfg)


class CusGuess(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    n: PositiveInt
    Xg: cs.DM | None = None
    Ug: cs.DM | None = None
    tg: cs.DM | None = None
    t0g: float | None = Field(ge=0, default=0)
    tfg: float | None = Field(gt=0, default=1)

    @model_validator(mode="after")
    def check(self) -> "CusGuess":

        if self.Xg.shape[1] != self.n + 1:
            raise ValueError(
                f"State guess matrix colums dont match. Expected {self.n + 1} but received {self.Xg.shape[1]}"
            )

        if self.Ug.shape[1] != self.n:
            raise ValueError(
                f"Control guess matrix colums dont match. Expected {self.n} but received {self.Ug.shape[1]}"
            )

        if self.tg.shape != (1, self.n + 1):
            raise ValueError(
                f'Time guess vector shape don"t match. Expected {(1, self.n + 1)} but received {self.tg.shape[1]}'
            )

        if self.tfg <= self.t0g:
            raise ValueError(
                f"Final time guess must be greater than initial time guess but received t0g ({self.t0g}) > tfg ({self.tfg}) "
            )

        return self

    def set_guess(self, dis_ocp: DisOcp, nlp: cs.Opti) -> None:
        """
        set guess to NLP

        Parameters
        ----------
        dis_ocp : DisOcp
            discrete OCP
        nlp : cs.Opti
            Opti
        """

        for name, att in zip(["state", "control"], ["X", "U"]):
            if getattr(dis_ocp, att).shape != getattr(self, f"{att}g").shape:
                raise ValueError(
                    f"Guess for the {name} matrix has incorrect row dimension, expected {getattr(dis_ocp, att).shape} but received {getattr(self, f'{att}g').shape}"
                )

        nlp.set_initial(dis_ocp.X, self.Xg)
        nlp.set_initial(dis_ocp.U, self.Ug)

        if not isinstance(dis_ocp.t, cs.DM) or not dis_ocp.t.is_constant():
            for i in cs.symvar(dis_ocp.t):
                if i.shape == (1, self.n + 1):
                    nlp.set_initial(dis_ocp.t, self.tg)
                    break
                if i.shape == (1, 1):
                    if cs.is_equal(dis_ocp.t0, i):
                        nlp.set_initial(dis_ocp.t0, self.t0g)
                    elif cs.is_equal(dis_ocp.tf, i):
                        nlp.set_initial(dis_ocp.tf, self.tfg)
