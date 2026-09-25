"""
underspec_sim.core.params: Primitives and configuration parameters.
Matches symbols in strategic_underspecification.tex:
- k: attribute count
- g: assistant raw guessing accuracy in (0, 1)
- L: loss per unresolved wrong attribute
- c_Q: cost of answering one clarifying question (common across users)
- Lambda: L * (1 - g), expected loss from an unasked/guessed attribute
- V: gross value of task success
- mu_A: leader's weight on true downstream user welfare in (0, 1]
- lambda_A: leader's perceived friction cost per clarifying question
- U_bar: reservation utility for participation constraint (IR)
"""

from typing import Optional
from pydantic import BaseModel, Field, computed_field, model_validator


class ModelParams(BaseModel):
    """
    Primitives for the Stackelberg under-specification game.
    """
    k: float = Field(default=10.0, gt=0, description="Attribute count k")
    g: float = Field(default=0.5, gt=0.0, lt=1.0, description="Assistant guess accuracy g in (0, 1)")
    L: float = Field(default=10.0, gt=0.0, description="Loss per unresolved wrong attribute L")
    c_Q: float = Field(default=2.0, gt=0.0, description="Cost of answering one clarifying question c_Q")
    V: float = Field(default=100.0, gt=0.0, description="Gross value of task success V")
    mu_A: float = Field(default=1.0, gt=0.0, le=1.0, description="Leader weight on user welfare mu_A in (0, 1]")
    lambda_A: float = Field(default=2.0, gt=0.0, description="Leader perceived question cost lambda_A")
    U_bar: float = Field(default=0.0, description="Reservation utility U_bar for IR constraint")

    @computed_field
    @property
    def Lambda(self) -> float:
        r"""\Lambda \equiv L(1 - g): expected loss from an unasked/guessed attribute."""
        return self.L * (1.0 - self.g)

    @computed_field
    @property
    def kappa_star(self) -> float:
        r"""\kappa^* = (\Lambda + c_Q) / (2k): first-best bang-bang threshold (Proposition 3)."""
        return (self.Lambda + self.c_Q) / (2.0 * self.k)

    @computed_field
    @property
    def kappa_star_biased(self) -> float:
        r"""\kappa^{*B}: biased first-best threshold for leader (Section 6.3)."""
        # When mu_A is present, c_Q^eff = c_Q + (lambda_A - c_Q)/mu_A
        c_Q_eff = self.c_Q + (self.lambda_A - self.c_Q) / self.mu_A
        return (self.Lambda + c_Q_eff) / (2.0 * self.k)

    @property
    def is_unbiased(self) -> bool:
        """Leader is unbiased iff mu_A == 1.0 and lambda_A == c_Q."""
        return abs(self.mu_A - 1.0) < 1e-9 and abs(self.lambda_A - self.c_Q) < 1e-9

    def with_bias(self, lambda_A: float, mu_A: Optional[float] = None) -> "ModelParams":
        """Return a copy with updated leader bias parameters."""
        new_mu = self.mu_A if mu_A is None else mu_A
        return self.model_copy(update={"lambda_A": lambda_A, "mu_A": new_mu})
