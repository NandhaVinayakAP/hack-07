"""REUSABLE DECISION PATTERN: travel-booking argument-grounding guardrail  (#guardrails)

What it does
    Before the travel agent runs any tool — search or booking — Strands Decider 2B checks the
    proposed tool call against the full conversation:

    Search tools (search_flights, search_hotels):
    - are the core search arguments (origin/destination city, travel dates, traveller/guest count)
      grounded in what the user actually said, not invented or assumed by the LLM?
    If any search argument is missing the agent is sent back to ask a focused follow-up question
    instead of firing a search with invented values.

    Booking tools (book_flight, book_hotel):
    - same argument-grounding check as for search; plus
    - is the user clearly asking to complete a booking now, rather than asking about prices or
      availability?
    Fully grounded and explicitly requested bookings run; otherwise the agent is sent back with
    targeted feedback so it asks the user for the missing detail instead of guessing.
    When the decider is unsure (confidence in the unsure band) a human confirms.

Typed questions and options
    search_grounded (noul)  -- were origin/destination, dates and traveller count stated for a search?
    args_grounded (noul)    -- were destination, dates and traveller count all stated for a booking?
    explicit_request (noul) -- is the user's latest message a booking instruction, not a query?
    See QUESTIONS below.

Wiring
    `TravelBookingGuard` is a Strands `InterventionHandler`; agent/agent.py passes it to the Agent:
        Agent(..., interventions=[TravelBookingGuard(decider)])
    It guards ALL_TOOLS; BOOKING_TOOLS receive the full three-question check; SEARCH_TOOLS receive
    the single search_grounded question.
    If the agent retries a call that was just sent back unchanged, the retry is denied (Strands
    requires guide handlers to converge).

How to reuse it
    Copy this file, set BOOKING_TOOLS and SEARCH_TOOLS to the irreversible and search tools in
    your domain, and reword the questions for the arguments that must be grounded.
    Thresholds are in `before_tool_call`.
"""

from __future__ import annotations

import json
from typing import Any

from strands.interventions import Confirm, Deny, Guide, InterventionHandler, Proceed

from decider_kit import Decider
from decider_kit.patterns.guardrails import render_messages

PATTERN_TAG = "#guardrails"

SEARCH_TOOLS = {"search_flights", "search_hotels"}
BOOKING_TOOLS = {"book_flight", "book_hotel"}
ALL_GUARDED_TOOLS = SEARCH_TOOLS | BOOKING_TOOLS

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

QUESTIONS = {
    "search_grounded": Decider.noul(
        "Are all the search arguments in the proposed tool call — origin and/or destination city, "
        "travel dates or date range, and number of travellers or guests — explicitly stated by the "
        "user in the conversation, rather than assumed or filled in by the assistant?",
        {
            "true": "the user mentioned every required search argument in their own words: "
                    "the city or cities, the date(s), and the traveller or guest count",
            "false": "at least one search argument (a city, a date, or the traveller/guest count) "
                     "was not said by the user and was assumed, guessed or invented by the assistant",
        },
    ),
    "args_grounded": Decider.noul(
        "Are the destination, travel dates, and number of travellers or guests in the proposed "
        "booking explicitly stated by the user in the conversation, rather than assumed or filled "
        "in by the assistant?",
        {
            "true": "the user stated the destination city, the specific date or date range, "
                    "and the number of travellers or guests in their own words",
            "false": "at least one of destination, dates or traveller count was not said by the user "
                     "and was assumed, guessed or invented by the assistant",
        },
    ),
    "explicit_request": Decider.noul(
        "Is the user's most recent message a direct instruction to complete the booking now, "
        "rather than a question about options, prices or availability?",
        {
            "true": "an instruction such as 'book it', 'go ahead', 'reserve that flight', "
                    "'book us two rooms', or naming a specific flight/hotel to confirm",
            "false": "a question ('what are my options?', 'how much is it?', 'when does it depart?'), "
                     "a search request, a price check, or a statement that doesn't request confirmation",
        },
    ),
}

# The search check only needs search_grounded; booking needs all three.
_SEARCH_QUESTIONS = {k: QUESTIONS[k] for k in ("search_grounded",)}
_BOOKING_QUESTIONS = {k: QUESTIONS[k] for k in ("args_grounded", "explicit_request")}


# ---------------------------------------------------------------------------
# Intervention handler
# ---------------------------------------------------------------------------

class TravelBookingGuard(InterventionHandler):
    """Intervention handler that guards all travel tools with Strands Decider 2B argument-grounding checks.

    Search tools are guarded by `search_grounded` only — if the agent is about to fire a search
    with an invented city or fabricated date the guard sends it back to ask the user.

    Booking tools run the full check: `args_grounded` (was every booking argument stated?) and
    `explicit_request` (is the user actually asking to book right now?).
    """

    name = "travel-booking-guard"

    def __init__(
        self,
        decider: Decider,
        yes: float = 0.5,
        unsure_band: tuple[float, float] = (0.4, 0.6),
    ):
        self.decider = decider
        self.yes = yes
        self.unsure_band = unsure_band
        self.log: list[dict[str, Any]] = []
        self._guided: set[str] = set()  # calls already sent back; same call again is denied

    def before_invocation(self, event, **kwargs):
        self._guided.clear()  # new user message: previously guided calls may now be justified
        return Proceed()

    def before_tool_call(self, event, **kwargs):  # noqa: PLR0911
        name = event.tool_use["name"]
        if name not in ALL_GUARDED_TOOLS:
            return Proceed()

        args = event.tool_use.get("input") or {}
        sig = f"{name}:{json.dumps(args, sort_keys=True)}"

        if sig in self._guided:
            # Guide must converge: same call unchanged is denied
            self.log.append({
                "pattern": PATTERN_TAG,
                "tool": name,
                "args": args,
                "verdict": "deny",
                "reason": "retried a guided call unchanged",
            })
            return Deny(
                reason="This exact call was already rejected. "
                       "Ask the user for the missing detail (city, dates, or traveller count) instead."
            )

        state = (
            "A travel-booking conversation, then a tool call the assistant wants to make.\n\n"
            f"--- CONVERSATION ---\n{render_messages(event.agent.messages)}\n\n"
            f"--- PROPOSED TOOL CALL ---\ntool: {name}\narguments: {json.dumps(args)}"
        )

        # ---- Search tools: one question -----------------------------------------------
        if name in SEARCH_TOOLS:
            return self._check_search(name, args, sig, state)

        # ---- Booking tools: two questions ---------------------------------------------
        return self._check_booking(name, args, sig, state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_search(
        self, name: str, args: dict, sig: str, state: str
    ):
        a = self.decider.ask(state, _SEARCH_QUESTIONS)
        s = a["search_grounded"].p_yes or 0.0

        rec: dict[str, Any] = {
            "pattern": PATTERN_TAG,
            "tool": name,
            "args": args,
            "search_grounded": round(s, 3),
            "decider_ms": round(self.decider.last_latency_ms or 0),
        }

        if s < self.yes:
            rec["verdict"] = "guide"
            self._guided.add(sig)
            self.log.append(rec)
            return Guide(
                feedback=(
                    "One or more search arguments (origin, destination, travel dates, or number of "
                    "travellers/guests) were not stated by the user. Ask a focused follow-up "
                    "question to collect the missing detail before searching."
                )
            )

        rec["verdict"] = "proceed"
        self.log.append(rec)
        return Proceed()

    def _check_booking(
        self, name: str, args: dict, sig: str, state: str
    ):
        a = self.decider.ask(state, _BOOKING_QUESTIONS)
        g = a["args_grounded"].p_yes or 0.0
        e = a["explicit_request"].p_yes or 0.0

        rec: dict[str, Any] = {
            "pattern": PATTERN_TAG,
            "tool": name,
            "args": args,
            "args_grounded": round(g, 3),
            "explicit_request": round(e, 3),
            "decider_ms": round(self.decider.last_latency_ms or 0),
        }

        lo, hi = self.unsure_band

        if g < self.yes:
            rec["verdict"] = "guide"
            self._guided.add(sig)
            self.log.append(rec)
            return Guide(
                feedback=(
                    "One or more booking arguments (destination, travel date(s), or number of "
                    "travellers/guests) were not stated by the user. Ask a focused follow-up "
                    "question to collect the missing detail before booking."
                )
            )

        if lo <= g <= hi:
            # Grounding is uncertain: ask a human to confirm rather than guess
            rec["verdict"] = "confirm"
            self.log.append(rec)
            return Confirm(
                prompt=f"Approve {name} {json.dumps(args)}? (grounding confidence: {round(g, 2)})",
                response="yes",
            )

        if e < self.yes:
            rec["verdict"] = "guide"
            self._guided.add(sig)
            self.log.append(rec)
            return Guide(
                feedback=(
                    "The user has not asked to complete this booking. "
                    "Answer their question or show the search results, then offer to book."
                )
            )

        rec["verdict"] = "proceed"
        self.log.append(rec)
        return Proceed()
