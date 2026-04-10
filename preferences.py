from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spice.protocols import WorldState

_VALID_VIBES = {"professional", "casual", "urgent", "reflective"}
_DEFAULT_VIBE = "professional"


def get_user_vibe(state: "WorldState | None" = None) -> str:
    """Return the user's current vibe derived from WorldState.

    Priority order:
    1. Explicit ``domain_state["user_vibe"]`` override.
    2. Vibe inferred from the urgency of the active personal entity.
    3. Vibe inferred from the highest-priority signal tagged ``vibe``.
    4. Default: ``"professional"``.
    """
    if state is None:
        return _DEFAULT_VIBE

    # 1. Explicit override stored in domain_state
    raw = state.domain_state.get("user_vibe")
    if isinstance(raw, str) and raw in _VALID_VIBES:
        return raw

    # 2. Infer from personal entity urgency
    entity = _personal_entity(state)
    urgency = _as_str(entity.get("urgency"))
    if urgency == "high":
        return "urgent"

    # 3. Infer from signals tagged as vibe signals
    for signal in state.signals:
        if not isinstance(signal, dict):
            continue
        if signal.get("tag") == "vibe" or signal.get("type") == "vibe":
            value = _as_str(signal.get("value") or signal.get("label"))
            if value in _VALID_VIBES:
                return value

    return _DEFAULT_VIBE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _personal_entity(state: "WorldState") -> dict[str, Any]:
    entity = state.entities.get("personal.assistant.current")
    if isinstance(entity, dict):
        return entity
    return {}


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return ""
