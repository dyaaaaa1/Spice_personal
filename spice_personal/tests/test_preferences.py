"""Tests for preferences.get_user_vibe dynamic vibe detection."""
import unittest

from spice.protocols import WorldState

from preferences import get_user_vibe


class TestGetUserVibeNoState(unittest.TestCase):
    def test_returns_professional_when_state_is_none(self) -> None:
        self.assertEqual(get_user_vibe(None), "professional")

    def test_returns_professional_with_no_args(self) -> None:
        self.assertEqual(get_user_vibe(), "professional")


class TestGetUserVibeDomainStateOverride(unittest.TestCase):
    def test_returns_domain_state_vibe_when_valid(self) -> None:
        state = WorldState(id="s1", domain_state={"user_vibe": "casual"})
        self.assertEqual(get_user_vibe(state), "casual")

    def test_ignores_invalid_domain_state_vibe(self) -> None:
        state = WorldState(id="s2", domain_state={"user_vibe": "galaxy_brain"})
        self.assertEqual(get_user_vibe(state), "professional")

    def test_ignores_non_string_domain_state_vibe(self) -> None:
        state = WorldState(id="s3", domain_state={"user_vibe": 42})
        self.assertEqual(get_user_vibe(state), "professional")

    def test_all_valid_vibes_accepted(self) -> None:
        for vibe in ("professional", "casual", "urgent", "reflective"):
            with self.subTest(vibe=vibe):
                state = WorldState(id=f"s-{vibe}", domain_state={"user_vibe": vibe})
                self.assertEqual(get_user_vibe(state), vibe)


class TestGetUserVibeUrgencyInference(unittest.TestCase):
    def test_high_urgency_entity_returns_urgent(self) -> None:
        state = WorldState(
            id="s4",
            entities={
                "personal.assistant.current": {
                    "latest_question": "Should I quit?",
                    "urgency": "high",
                }
            },
        )
        self.assertEqual(get_user_vibe(state), "urgent")

    def test_normal_urgency_entity_falls_through_to_default(self) -> None:
        state = WorldState(
            id="s5",
            entities={
                "personal.assistant.current": {
                    "urgency": "normal",
                }
            },
        )
        self.assertEqual(get_user_vibe(state), "professional")

    def test_domain_state_overrides_urgency(self) -> None:
        state = WorldState(
            id="s6",
            domain_state={"user_vibe": "reflective"},
            entities={
                "personal.assistant.current": {"urgency": "high"}
            },
        )
        self.assertEqual(get_user_vibe(state), "reflective")


class TestGetUserVibeSignalInference(unittest.TestCase):
    def test_vibe_signal_by_tag(self) -> None:
        state = WorldState(
            id="s7",
            signals=[{"tag": "vibe", "value": "casual"}],
        )
        self.assertEqual(get_user_vibe(state), "casual")

    def test_vibe_signal_by_type(self) -> None:
        state = WorldState(
            id="s8",
            signals=[{"type": "vibe", "label": "reflective"}],
        )
        self.assertEqual(get_user_vibe(state), "reflective")

    def test_unrelated_signals_ignored(self) -> None:
        state = WorldState(
            id="s9",
            signals=[{"tag": "risk", "value": "casual"}],
        )
        self.assertEqual(get_user_vibe(state), "professional")

    def test_invalid_vibe_signal_value_ignored(self) -> None:
        state = WorldState(
            id="s10",
            signals=[{"tag": "vibe", "value": "chaotic_neutral"}],
        )
        self.assertEqual(get_user_vibe(state), "professional")

    def test_urgency_takes_priority_over_signal(self) -> None:
        state = WorldState(
            id="s11",
            entities={"personal.assistant.current": {"urgency": "high"}},
            signals=[{"tag": "vibe", "value": "casual"}],
        )
        # urgency=high → "urgent" wins over signal-derived "casual"
        self.assertEqual(get_user_vibe(state), "urgent")


class TestGetUserVibeEmptyState(unittest.TestCase):
    def test_empty_world_state_returns_default(self) -> None:
        state = WorldState(id="s12")
        self.assertEqual(get_user_vibe(state), "professional")


if __name__ == "__main__":
    unittest.main()
