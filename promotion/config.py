"""Explicit, auditable configuration for the promotion gate (Phase 9, ADR-014).

The gate margins — the F2P gain the challenger must clear and the largest P2P
drop tolerated — are configuration, not ambient environment: they are resolved
once here and threaded explicitly into ``rules.decide`` and the decision record.
An override therefore appears *inside* the audit trail (the recorded
``thresholds`` plus ``overridden``), so a weakened gate is visible after the
fact instead of being applied silently at import time.

The absolute quality floors stay hardcoded literals in ``promotion.rules``: a
candidate below a floor can never be promoted regardless of this config.

Follows the repo's ``EvalConfig``/``DataPipelineConfig`` pattern
(pydantic-settings: ``.env`` + ``PROMOTE_*`` env vars > defaults). The env var
names are unchanged from the previous implementation so an existing CI override
keeps working — it is now explicit and recorded rather than ambient.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from promotion.rules import PROMOTE_MAX_P2P_REGRESSION, PROMOTE_MIN_F2P_GAIN

__all__ = ["PromotionConfig"]


class PromotionConfig(BaseSettings):
    """Promotion gate margins; ``PROMOTE_*`` env / ``.env`` override the defaults.

    Defaults (0.05 gain, 0.02 regression) reproduce the historical gate, so
    behaviour is unchanged when nothing is configured.
    """

    min_f2p_gain: float = PROMOTE_MIN_F2P_GAIN
    max_p2p_regression: float = PROMOTE_MAX_P2P_REGRESSION

    model_config = SettingsConfigDict(
        env_prefix="PROMOTE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    def overridden(self) -> list[str]:
        """Field names set from outside the defaults (env/init), sorted.

        Recorded in the decision record so an operator can see which margin was
        overridden; empty when the gate ran on the defaults.
        """
        return sorted(self.model_fields_set)
