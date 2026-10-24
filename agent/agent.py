"""The travel-booking agent: an LLM does the talking (Amazon Bedrock), Strands Decider 2B guards bookings.

Every call to book_flight or book_hotel passes through TravelBookingGuard first.
If any booking argument (destination, dates, traveller count) is not grounded in the
conversation the agent is sent back to ask a follow-up question instead of booking.

    from agent.agent import HackathonAgent
    out = HackathonAgent().chat("I'd like to fly to London on November 1st for 2 people.")
    out["reply"], out["decisions"]
"""

from __future__ import annotations

import time
from typing import Any

from strands import Agent

from decider_kit import Decider
from decider_kit.llm import agent_model, default_model_id, image_blocks, strip_thinking

from .decision import TravelBookingGuard
from .tools import TOOLS

NAME = "Travel-booking agent with Strands Decider 2B argument-grounding guardrail"

SYSTEM_PROMPT = """You are a travel-booking assistant. Help users search for and book flights and hotels.

Rules:
- Use search_flights and search_hotels to show options before offering to book.
- Use book_flight and book_hotel only when the user has explicitly asked to complete a booking.
- Never assume a destination, date, or traveller count. If any of these is missing, ask a
  focused follow-up question instead of guessing.
- Dates must be in YYYY-MM-DD format; infer the year only if the user mentions a month and day
  and the context makes it unambiguous.
- Be brief and friendly. Confirm booking details back to the user before proceeding when
  there is any ambiguity."""

# Demos only: an eager prompt makes the agent act without asking, so the decision has mistakes
# to catch on camera. The intro slide says so. Set eager=False to show normal behaviour.
EAGER_PROMPT = """You are a travel-booking assistant that acts immediately without asking questions.
Use the most plausible destination, dates and traveller count from context (default: 1 traveller,
today's date, the destination mentioned). Call book_flight or book_hotel right away."""


class HackathonAgent:
    """One Strands agent per session, with TravelBookingGuard wired in."""

    def __init__(
        self,
        decider: Decider | None = None,
        model_id: str | None = None,
        eager: bool | None = None,
    ):
        import os

        self.decider = decider or Decider()
        self.eager = eager if eager is not None else os.environ.get("DEMO_EAGER") == "1"
        self.model_id = model_id or default_model_id()
        self.sessions: dict[str, tuple[Agent, TravelBookingGuard]] = {}

    def _session(self, session_id: str) -> tuple[Agent, TravelBookingGuard]:
        if session_id not in self.sessions:
            guard = TravelBookingGuard(self.decider)
            agent = Agent(
                model=agent_model(self.model_id),
                system_prompt=EAGER_PROMPT if self.eager else SYSTEM_PROMPT,
                tools=TOOLS,
                interventions=[guard],
                callback_handler=None,
            )
            self.sessions[session_id] = (agent, guard)
        return self.sessions[session_id]

    def chat(
        self,
        message: str,
        session_id: str = "default",
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """`images` (optional): photos attached by the user (paths, bytes or base64)."""
        agent, guard = self._session(session_id)
        before = len(guard.log)
        t0 = time.perf_counter()
        result = agent(
            [*image_blocks(images), {"text": message}] if images else message,
            invocation_state={"images": list(images or [])},
        )
        reply = strip_thinking(str(result))
        return {
            "reply": reply,
            "decisions": guard.log[before:],
            "model_id": self.model_id,
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "agent": NAME,
            "eager": self.eager,
        }


if __name__ == "__main__":
    import json
    import sys

    a = HackathonAgent()
    prompts = sys.argv[1:] or [
        "I want to fly to London and book a hotel.",
        "I'm flying from New York. The trip is November 1st to November 5th, just me.",
        "Book flight FL001 and a room at the Grand Kensington please.",
    ]
    for msg in prompts:
        print(json.dumps(a.chat(msg), indent=1))
