"""Validation suite for the travel-booking #guardrails agent.

Five independent groups, one per requirement:

1. Search validation    — a search call with uninvented/ungrounded args is blocked;
                          a fully-grounded search proceeds.
2. Booking validation   — original booking cases produce the right verdicts without regression.
3. Convergence guard    — the same denied call retried unchanged is Denied for both search
                          and booking tools.
4. Eval integrity       — every expected key in eval_cases.jsonl is a real QUESTIONS key;
                          every state uses the runtime text layout; splits are tune/test only.
5. Demo consistency     — the five demo turns use agent-facing tool names and describe the
                          runtime guardrail behaviour correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fakes import FakeDecider
from strands.interventions import Deny, Guide, Proceed

from agent.decision import (
    ALL_GUARDED_TOOLS,
    BOOKING_TOOLS,
    PATTERN_TAG,
    QUESTIONS,
    SEARCH_TOOLS,
    TravelBookingGuard,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "agent" / "eval_cases.jsonl"
DEMO_PATH = ROOT / "agent" / "demo_script.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guard(rule) -> TravelBookingGuard:
    """Build a TravelBookingGuard backed by a FakeDecider with the given rule."""
    return TravelBookingGuard(FakeDecider(rule))


def _event(tool_name: str, args: dict, messages: list | None = None) -> SimpleNamespace:
    """Minimal Strands tool-call event as used in before_tool_call."""
    msgs = messages or [{"role": "user", "content": [{"text": "some user message"}]}]
    return SimpleNamespace(
        tool_use={"name": tool_name, "input": args},
        agent=SimpleNamespace(messages=msgs),
    )


def _invocation_event(messages: list | None = None) -> SimpleNamespace:
    """Minimal invocation event as used in before_invocation."""
    msgs = messages or [{"role": "user", "content": [{"text": "hello"}]}]
    return SimpleNamespace(messages=msgs)


# ---------------------------------------------------------------------------
# 1. Search validation
# ---------------------------------------------------------------------------

class TestSearchValidation:
    """A search_flights or search_hotels call with ungrounded args must be blocked."""

    def test_search_flights_ungrounded_is_guided(self):
        # Decider says search_grounded=0.1 (not grounded)
        guard = _guard(lambda s, n, q: 0.1 if n == "search_grounded" else 0.9)
        result = guard.before_tool_call(
            _event("search_flights", {"origin": "New York", "destination": "London",
                                      "date": "2025-11-01", "travellers": 1})
        )
        assert isinstance(result, Guide), "ungrounded search_flights should produce Guide"
        assert guard.log[-1]["verdict"] == "guide"
        assert guard.log[-1]["tool"] == "search_flights"
        assert "search_grounded" in guard.log[-1]

    def test_search_hotels_ungrounded_is_guided(self):
        guard = _guard(lambda s, n, q: 0.1 if n == "search_grounded" else 0.9)
        result = guard.before_tool_call(
            _event("search_hotels", {"city": "Paris", "check_in": "2025-11-02",
                                     "check_out": "2025-11-05", "guests": 1})
        )
        assert isinstance(result, Guide)
        assert guard.log[-1]["verdict"] == "guide"

    def test_search_flights_missing_all_three_fields_is_guided(self):
        """Decider sees a state where origin, date, and traveller count are all absent."""
        guard = _guard(lambda s, n, q: 0.05)  # always ungrounded
        result = guard.before_tool_call(
            _event(
                "search_flights",
                {"origin": "New York", "destination": "London",
                 "date": "2025-11-01", "travellers": 1},
                messages=[{"role": "user", "content": [{"text": "I want to go to London."}]}],
            )
        )
        assert isinstance(result, Guide)

    def test_search_flights_grounded_proceeds(self):
        # Decider says search_grounded=0.95 (fully grounded)
        guard = _guard(lambda s, n, q: 0.95 if n == "search_grounded" else 0.9)
        result = guard.before_tool_call(
            _event(
                "search_flights",
                {"origin": "New York", "destination": "London",
                 "date": "2025-11-01", "travellers": 1},
                messages=[{"role": "user", "content": [
                    {"text": "I'm flying from New York to London on November 1st, just me."}
                ]}],
            )
        )
        assert isinstance(result, Proceed)
        assert guard.log[-1]["verdict"] == "proceed"

    def test_search_hotels_grounded_proceeds(self):
        guard = _guard(lambda s, n, q: 0.95)
        result = guard.before_tool_call(
            _event("search_hotels", {"city": "Tokyo", "check_in": "2025-11-05",
                                     "check_out": "2025-11-08", "guests": 2})
        )
        assert isinstance(result, Proceed)

    def test_guide_feedback_mentions_missing_fields(self):
        """The Guide feedback must tell the agent what was missing."""
        guard = _guard(lambda s, n, q: 0.1)
        result = guard.before_tool_call(
            _event("search_flights", {"origin": "X", "destination": "Y",
                                      "date": "2025-11-01", "travellers": 1})
        )
        assert isinstance(result, Guide)
        fb = result.feedback.lower()
        # Must mention something about the missing pieces
        assert any(word in fb for word in ("origin", "destination", "date", "traveller", "search")), (
            f"feedback should name the missing search arguments; got: {result.feedback!r}"
        )

    def test_unguarded_tool_skipped(self):
        """Tools not in ALL_GUARDED_TOOLS must pass through without a decider call."""
        guard = _guard(lambda s, n, q: 0.0)  # would block everything
        result = guard.before_tool_call(_event("get_exchange_rate", {"currency": "EUR"}))
        assert isinstance(result, Proceed)
        assert guard.log == []  # no decider call at all

    def test_state_text_contains_conversation_and_tool_call(self):
        """The state string passed to the decider must include both sections."""
        captured: list[str] = []

        class CapturingFakeDecider(FakeDecider):
            def ask(self, state, questions, images=None):
                captured.append(state)
                return super().ask(state, questions, images)

        guard = TravelBookingGuard(CapturingFakeDecider(lambda s, n, q: 0.9))
        guard.before_tool_call(
            _event("search_flights",
                   {"origin": "New York", "destination": "London",
                    "date": "2025-11-01", "travellers": 1},
                   messages=[{"role": "user", "content": [{"text": "fly from New York"}]}])
        )
        assert captured, "decider was never called"
        state = captured[0]
        assert "--- CONVERSATION ---" in state
        assert "--- PROPOSED TOOL CALL ---" in state
        assert "search_flights" in state

    @pytest.mark.parametrize("tool", sorted(SEARCH_TOOLS))
    def test_search_questions_asked_not_booking_questions(self, tool):
        """Search tools must only call the decider with search_grounded, never args_grounded."""
        asked: list[dict] = []

        class RecordingFakeDecider(FakeDecider):
            def ask(self, state, questions, images=None):
                asked.append(dict(questions))
                return super().ask(state, questions, images)

        guard = TravelBookingGuard(RecordingFakeDecider(lambda s, n, q: 0.9))
        guard.before_tool_call(_event(tool, {"city": "X", "check_in": "2025-11-01",
                                             "check_out": "2025-11-03", "guests": 1}))
        assert asked, "decider never called"
        assert set(asked[0].keys()) == {"search_grounded"}, (
            f"search tool should only ask search_grounded; got {set(asked[0].keys())}"
        )


# ---------------------------------------------------------------------------
# 2. Booking validation — no regressions on original booking cases
# ---------------------------------------------------------------------------

class TestBookingValidation:
    """Original booking-tool behaviour must be fully preserved."""

    def test_grounded_explicit_booking_proceeds(self):
        # Both args_grounded=0.9 and explicit_request=0.9
        guard = _guard(lambda s, n, q: 0.9)
        result = guard.before_tool_call(
            _event("book_flight", {"flight_id": "FL001", "travellers": 2,
                                   "passenger_name": "Alice"})
        )
        assert isinstance(result, Proceed)
        assert guard.log[-1]["verdict"] == "proceed"

    def test_ungrounded_booking_is_guided(self):
        # args_grounded=0.1 → Guide even if explicit_request is high
        guard = _guard(lambda s, n, q: 0.1 if n == "args_grounded" else 0.9)
        result = guard.before_tool_call(
            _event("book_flight", {"flight_id": "FL001", "travellers": 1,
                                   "passenger_name": "Bob"})
        )
        assert isinstance(result, Guide)
        assert guard.log[-1]["verdict"] == "guide"

    def test_grounded_but_not_explicit_is_guided(self):
        # args_grounded=0.9 but explicit_request=0.1 → Guide
        guard = _guard(lambda s, n, q: 0.1 if n == "explicit_request" else 0.9)
        result = guard.before_tool_call(
            _event("book_hotel", {"hotel_id": "HT001", "check_in": "2025-11-01",
                                  "check_out": "2025-11-05", "guests": 2,
                                  "guest_name": "Carol"})
        )
        assert isinstance(result, Guide)
        assert guard.log[-1]["verdict"] == "guide"

    def test_hotel_grounded_explicit_proceeds(self):
        guard = _guard(lambda s, n, q: 0.9)
        result = guard.before_tool_call(
            _event("book_hotel", {"hotel_id": "HT003", "check_in": "2025-11-02",
                                  "check_out": "2025-11-05", "guests": 1,
                                  "guest_name": "Dana"})
        )
        assert isinstance(result, Proceed)

    def test_unsure_grounding_triggers_confirm(self):
        # args_grounded=0.5 — inside default unsure_band (0.4, 0.6)
        from strands.interventions import Confirm
        guard = _guard(lambda s, n, q: 0.5 if n == "args_grounded" else 0.9)
        result = guard.before_tool_call(
            _event("book_flight", {"flight_id": "FL002", "travellers": 1,
                                   "passenger_name": "Eve"})
        )
        assert isinstance(result, Confirm)
        assert guard.log[-1]["verdict"] == "confirm"

    def test_booking_questions_asked_not_search_questions(self):
        """Booking tools must call the decider with args_grounded + explicit_request, not search_grounded."""
        asked: list[dict] = []

        class RecordingFakeDecider(FakeDecider):
            def ask(self, state, questions, images=None):
                asked.append(dict(questions))
                return super().ask(state, questions, images)

        guard = TravelBookingGuard(RecordingFakeDecider(lambda s, n, q: 0.9))
        guard.before_tool_call(
            _event("book_flight", {"flight_id": "FL001", "travellers": 1,
                                   "passenger_name": "Frank"})
        )
        assert asked
        assert set(asked[0].keys()) == {"args_grounded", "explicit_request"}, (
            f"booking tool should ask args_grounded + explicit_request; got {set(asked[0].keys())}"
        )

    def test_log_records_decider_ms(self):
        guard = _guard(lambda s, n, q: 0.9)
        guard.before_tool_call(
            _event("book_flight", {"flight_id": "FL001", "travellers": 1,
                                   "passenger_name": "Grace"})
        )
        assert "decider_ms" in guard.log[-1]
        assert isinstance(guard.log[-1]["decider_ms"], int)

    def test_pattern_tag_in_log(self):
        guard = _guard(lambda s, n, q: 0.9)
        guard.before_tool_call(
            _event("book_hotel", {"hotel_id": "HT001", "check_in": "2025-11-01",
                                  "check_out": "2025-11-03", "guests": 1,
                                  "guest_name": "Hal"})
        )
        assert guard.log[-1]["pattern"] == PATTERN_TAG == "#guardrails"


# ---------------------------------------------------------------------------
# 3. Convergence guard
# ---------------------------------------------------------------------------

class TestConvergenceGuard:
    """Retrying the same rejected call unchanged must always produce Deny."""

    def _ungrounded_guard(self) -> TravelBookingGuard:
        return _guard(lambda s, n, q: 0.1)  # every question fails grounding

    def test_search_retry_is_denied(self):
        guard = self._ungrounded_guard()
        ev = _event("search_flights", {"origin": "NYC", "destination": "LHR",
                                       "date": "2025-11-01", "travellers": 1})
        first = guard.before_tool_call(ev)
        second = guard.before_tool_call(ev)
        assert isinstance(first, Guide), "first call should be guided"
        assert isinstance(second, Deny), "retry of same search call should be denied"
        assert guard.log[-1]["verdict"] == "deny"
        assert "retried" in guard.log[-1]["reason"].lower()

    def test_booking_retry_is_denied(self):
        guard = self._ungrounded_guard()
        ev = _event("book_flight", {"flight_id": "FL001", "travellers": 1,
                                    "passenger_name": "Ivy"})
        first = guard.before_tool_call(ev)
        second = guard.before_tool_call(ev)
        assert isinstance(first, Guide)
        assert isinstance(second, Deny)

    def test_hotel_search_retry_is_denied(self):
        guard = self._ungrounded_guard()
        ev = _event("search_hotels", {"city": "Tokyo", "check_in": "2025-11-05",
                                      "check_out": "2025-11-08", "guests": 2})
        assert isinstance(guard.before_tool_call(ev), Guide)
        assert isinstance(guard.before_tool_call(ev), Deny)

    def test_hotel_booking_retry_is_denied(self):
        guard = self._ungrounded_guard()
        ev = _event("book_hotel", {"hotel_id": "HT001", "check_in": "2025-11-01",
                                   "check_out": "2025-11-03", "guests": 1,
                                   "guest_name": "Jack"})
        assert isinstance(guard.before_tool_call(ev), Guide)
        assert isinstance(guard.before_tool_call(ev), Deny)

    def test_new_invocation_resets_guided_set(self):
        """before_invocation must clear _guided so the same call is fresh after a new user turn."""
        guard = self._ungrounded_guard()
        ev = _event("search_flights", {"origin": "NYC", "destination": "LHR",
                                       "date": "2025-11-01", "travellers": 1})
        # First user turn: call is guided
        assert isinstance(guard.before_tool_call(ev), Guide)
        # Simulate a new user message arriving
        guard.before_invocation(_invocation_event())
        # Same call is now back to Guide (not Deny) after the reset
        assert isinstance(guard.before_tool_call(ev), Guide), (
            "after before_invocation, the same call should be Guide again, not Deny"
        )

    def test_different_args_not_denied(self):
        """A call with different args is a new call; it should not be immediately denied."""
        guard = self._ungrounded_guard()
        ev1 = _event("search_flights", {"origin": "NYC", "destination": "LHR",
                                        "date": "2025-11-01", "travellers": 1})
        ev2 = _event("search_flights", {"origin": "NYC", "destination": "CDG",
                                        "date": "2025-11-02", "travellers": 2})
        assert isinstance(guard.before_tool_call(ev1), Guide)
        # Different args — should be Guide again, not Deny
        assert isinstance(guard.before_tool_call(ev2), Guide), (
            "a search with different args should be guided, not denied as a retry"
        )

    def test_deny_log_entry_has_expected_fields(self):
        guard = self._ungrounded_guard()
        ev = _event("search_flights", {"origin": "A", "destination": "B",
                                       "date": "2025-11-01", "travellers": 1})
        guard.before_tool_call(ev)
        guard.before_tool_call(ev)
        deny_entry = guard.log[-1]
        assert deny_entry["verdict"] == "deny"
        assert deny_entry["tool"] == "search_flights"
        assert "reason" in deny_entry
        assert deny_entry["pattern"] == "#guardrails"


# ---------------------------------------------------------------------------
# 4. Evaluation integrity
# ---------------------------------------------------------------------------

class TestEvalIntegrity:
    """eval_cases.jsonl must be consistent with the runtime QUESTIONS and state layout."""

    def _cases(self) -> list[dict]:
        return [
            json.loads(line)
            for line in EVAL_PATH.read_text().splitlines()
            if line.strip()
        ]

    def test_all_expected_keys_are_question_names(self):
        for case in self._cases():
            unknown = set(case["expected"]) - set(QUESTIONS)
            assert not unknown, (
                f"expected key(s) {unknown} not in QUESTIONS; "
                f"tool: {case.get('state', '')[:80]}"
            )

    def test_all_expected_values_are_yes_or_no(self):
        for case in self._cases():
            for k, v in case["expected"].items():
                assert v in ("yes", "no"), (
                    f"expected value {v!r} for key {k!r} must be 'yes' or 'no'"
                )

    def test_state_uses_runtime_text_layout(self):
        """Every state must contain the two section headers that before_tool_call produces."""
        for case in self._cases():
            state = case["state"]
            assert "--- CONVERSATION ---" in state, (
                f"state missing '--- CONVERSATION ---' section:\n{state[:200]}"
            )
            assert "--- PROPOSED TOOL CALL ---" in state, (
                f"state missing '--- PROPOSED TOOL CALL ---' section:\n{state[:200]}"
            )

    def test_splits_are_tune_or_test(self):
        for case in self._cases():
            assert case["split"] in ("tune", "test"), (
                f"split {case['split']!r} is not 'tune' or 'test'"
            )

    def test_split_balance_is_reasonable(self):
        cases = self._cases()
        tune = sum(1 for c in cases if c["split"] == "tune")
        test = sum(1 for c in cases if c["split"] == "test")
        assert tune >= 2, f"need at least 2 tune cases, got {tune}"
        assert test >= 2, f"need at least 2 test cases, got {test}"
        ratio = tune / max(test, 1)
        assert 0.5 <= ratio <= 2.0, (
            f"tune/test split {tune}/{test} is too skewed (ratio {ratio:.2f})"
        )

    def test_each_case_has_required_fields(self):
        for i, case in enumerate(self._cases()):
            assert "state" in case, f"case {i} missing 'state'"
            assert "expected" in case, f"case {i} missing 'expected'"
            assert "split" in case, f"case {i} missing 'split'"
            assert isinstance(case["state"], str) and case["state"].strip(), (
                f"case {i} has empty state"
            )

    def test_search_cases_use_search_grounded_key(self):
        """Cases for search_flights / search_hotels must use 'search_grounded', not 'args_grounded'."""
        for case in self._cases():
            state = case["state"]
            if "search_flights" in state or "search_hotels" in state:
                assert "search_grounded" in case["expected"], (
                    f"search tool case should use 'search_grounded'; got {case['expected']}"
                )

    def test_booking_cases_use_booking_keys(self):
        """Cases for book_flight / book_hotel must use 'args_grounded' and/or 'explicit_request'."""
        booking_keys = {"args_grounded", "explicit_request"}
        for case in self._cases():
            state = case["state"]
            if "book_flight" in state or "book_hotel" in state:
                assert set(case["expected"]) <= booking_keys, (
                    f"booking case expected keys {set(case['expected'])} not subset of {booking_keys}"
                )

    def test_no_duplicate_states(self):
        cases = self._cases()
        states = [c["state"] for c in cases]
        assert len(states) == len(set(states)), "eval_cases.jsonl contains duplicate state values"

    def test_at_least_one_case_per_guarded_tool(self):
        """Every tool in ALL_GUARDED_TOOLS must appear in at least one eval case."""
        cases = self._cases()
        for tool in ALL_GUARDED_TOOLS:
            covered = any(tool in c["state"] for c in cases)
            assert covered, f"no eval case covers tool '{tool}'"


# ---------------------------------------------------------------------------
# 5. Demo consistency
# ---------------------------------------------------------------------------

class TestDemoConsistency:
    """demo_script.yaml must describe behaviour that matches the runtime guardrail logic."""

    def _script(self) -> dict:
        return yaml.safe_load(DEMO_PATH.read_text())

    def test_script_has_exactly_five_turns(self):
        turns = self._script()["turns"]
        assert len(turns) == 5, f"expected 5 turns, got {len(turns)}"

    def test_every_turn_has_say_and_narrate(self):
        for i, turn in enumerate(self._script()["turns"]):
            assert "say" in turn and turn["say"].strip(), f"turn {i} missing 'say'"
            assert "narrate" in turn and str(turn["narrate"]).strip(), f"turn {i} missing 'narrate'"

    def test_first_turn_describes_missing_arguments(self):
        """Turn 1 shows the guardrail blocking a search with incomplete info."""
        narrate = str(self._script()["turns"][0]["narrate"]).lower()
        # Must describe what is absent — no origin, no date, no traveller count
        has_missing = any(word in narrate for word in (
            "no origin", "no date", "no traveller", "missing", "without", "gives no", "but gives no",
            "not provided", "absent", "ungrounded", "invented"
        ))
        assert has_missing, (
            f"Turn 1 narration should describe missing search arguments; got: {narrate!r}"
        )

    def test_second_turn_describes_grounded_search(self):
        """Turn 2 shows the user supplying the missing origin, date, and traveller count."""
        narrate = str(self._script()["turns"][1]["narrate"]).lower()
        has_grounded = any(word in narrate for word in (
            "origin", "date", "traveller", "provides", "grounded", "supplied", "adds", "now"
        ))
        assert has_grounded, (
            f"Turn 2 narration should describe grounded search arguments; got: {narrate!r}"
        )

    def test_third_turn_describes_explicit_booking(self):
        """Turn 3 shows an explicit, grounded booking confirmation."""
        narrate = str(self._script()["turns"][2]["narrate"]).lower()
        has_booking = any(word in narrate for word in (
            "book", "confirm", "explicit", "flight", "grounded", "names"
        ))
        assert has_booking, (
            f"Turn 3 narration should describe an explicit booking; got: {narrate!r}"
        )

    def test_fourth_turn_describes_incomplete_hotel_search(self):
        """Turn 4 shows a hotel search or booking with missing dates."""
        narrate = str(self._script()["turns"][3]["narrate"]).lower()
        has_hotel_issue = any(word in narrate for word in (
            "hotel", "date", "check-in", "check in", "no check", "without", "missing", "gives no"
        ))
        assert has_hotel_issue, (
            f"Turn 4 narration should describe incomplete hotel arguments; got: {narrate!r}"
        )

    def test_fifth_turn_describes_grounded_hotel_booking(self):
        """Turn 5 shows the user providing dates and a grounded hotel booking succeeding."""
        narrate = str(self._script()["turns"][4]["narrate"]).lower()
        has_success = any(word in narrate for word in (
            "grounded", "provides", "dates", "names", "hotel", "book"
        ))
        assert has_success, (
            f"Turn 5 narration should describe a grounded hotel booking; got: {narrate!r}"
        )

    def test_script_has_required_top_level_keys(self):
        script = self._script()
        for key in ("title", "pitch", "features", "eager", "turns", "outro"):
            assert key in script, f"demo_script.yaml missing top-level key '{key}'"

    def test_features_list_is_non_empty(self):
        assert len(self._script()["features"]) >= 1

    def test_narrate_never_claims_a_specific_decider_verdict(self):
        """make_video.py appends the actual verdict; narrate lines must not pre-claim it."""
        forbidden = ("decider allowed", "decider blocked", "decider approved",
                     "decider denied", "decider guided", "verdict:")
        for i, turn in enumerate(self._script()["turns"]):
            narrate = str(turn["narrate"]).lower()
            for phrase in forbidden:
                assert phrase not in narrate, (
                    f"Turn {i} narration claims a specific decider verdict ({phrase!r}); "
                    f"make_video.py appends the real verdict — remove it from narrate"
                )

    def test_say_fields_are_natural_user_messages(self):
        """say fields must be non-empty strings (they are typed into the demo UI)."""
        for i, turn in enumerate(self._script()["turns"]):
            say = turn["say"]
            assert isinstance(say, str) and len(say.strip()) > 5, (
                f"Turn {i} 'say' field is too short or empty: {say!r}"
            )
