# A travel-booking agent that cannot search or book unless you tell it where and when

By TODO your name

Booking the wrong flight is a classic LLM failure mode: the user says "I want to go to London"
and the agent invents a date, picks a seat count from thin air, and fires the booking before
anyone noticed a guess was made. This agent fixes that at the seam between the LLM and the
tool call, with a single forward pass of Strands Decider 2B before any tool runs.

## The project

The agent searches for and books flights and hotels in chat. It runs on Amazon Bedrock
(Amazon Nova Pro by default) via the Strands Agents SDK, deployed on Amazon Bedrock AgentCore
Runtime. What makes it different is what happens before any tool runs.

Every tool call -- search and booking -- is intercepted by TravelBookingGuard, a Strands
InterventionHandler that asks Strands Decider 2B one or two typed questions before letting the
tool proceed:

- Before search_flights or search_hotels: was every search argument (origin, destination,
  dates, traveller count) actually stated by the user?
- Before book_flight or book_hotel: were all booking arguments grounded in the conversation,
  and is the user's most recent message a direct instruction to book right now (not a price
  check or availability query)?

If any question comes back below the confidence threshold the agent is sent back with a
targeted feedback message so it asks the user for the missing detail. If the decider is unsure
(confidence in the 0.40-0.60 band) the call is routed to a human to confirm. Only when all
questions pass does the tool run.

A real conversation looks like this:

User: Find me a flight to London.
Agent: I need a few more details to search for flights. Could you tell me your departure
       city, your travel date, and how many people are travelling?
User: I'm flying from New York on November 1st, just me.
Agent: [searches] I found two flights: FL001 at $620 and FL002 at $780. Which would you like?
User: FL001 looks good, go ahead and book it.
Agent: [books] Done -- booking ref BK1001, New York to London on 2025-11-01 for 1 traveller, $620.

The first turn fires search_flights with invented values. The decider scores search_grounded
below 0.5 and guides the agent back. The second turn provides all three missing pieces; the
decider scores above 0.5 and the search runs. The third turn is an explicit booking instruction
with fully grounded arguments; both questions pass and the booking goes through.

< FIGURE 1 of 5: upload submission/article/img/demo.png here >
Alt text: The demo UI showing the conversation on the left and the Strands Decider 2B decision panel on the right, with search_grounded, args_grounded, explicit_request scores and verdict for each tool call.

## How I built it

The agent is built with the Strands Agents SDK. Four @tool functions in agent/tools.py hold
synthetic in-memory data for flights and hotels (no external API needed). The Strands Agent
in agent/agent.py wires the model, the tools, and the guardrail together:

    agent = Agent(
        model=agent_model(model_id),
        system_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        interventions=[TravelBookingGuard(decider)],
        callback_handler=None,
    )

Amazon Bedrock drives the conversation; the decider runs at the intervention seam before each
tool call. The demo UI is a FastAPI server with a single-page HTML front end that shows the
decision panel in real time. The whole thing deploys to Amazon Bedrock AgentCore Runtime with
one command.

I built the spec and implementation in Kiro using the Strands Decider Hackathon Kit from the
AWS User Group Madurai, which scaffolded the agent/, decider_kit/, and demo/ structure and
handled the AgentCore deployment.

## The decision I handed to Strands Decider 2B

The core problem is trust. The LLM is fluent and fast, but it will happily invent a departure
city or a date if the user left them out. Catching that with a prompt ("only book if the user
said the city") works sometimes; it fails on edge cases and produces a binary yes that carries
no calibrated confidence score you can branch on.

Strands Decider 2B returns a probability. The code can act on that number directly: proceed
above 0.5, guide below 0.5, and route to a human when it is between 0.4 and 0.6. There is no
second LLM call for the decision -- one forward pass of the 2B model at the intervention seam.

Three typed questions guard the four tools:

```python
QUESTIONS = {
    "search_grounded": Decider.noul(
        "Are all the search arguments in the proposed tool call -- origin and/or destination "
        "city, travel dates or date range, and number of travellers or guests -- explicitly "
        "stated by the user in the conversation, rather than assumed or filled in by the assistant?",
        {
            "true": "the user mentioned every required search argument in their own words: "
                    "the city or cities, the date(s), and the traveller or guest count",
            "false": "at least one search argument (a city, a date, or the traveller/guest count) "
                     "was not said by the user and was assumed, guessed or invented by the assistant",
        },
    ),
    "args_grounded": Decider.noul(
        "Are the destination, travel dates, and number of travellers or guests in the proposed "
        "booking explicitly stated by the user in the conversation, rather than assumed or "
        "filled in by the assistant?",
        {
            "true": "the user stated the destination city, the specific date or date range, "
                    "and the number of travellers or guests in their own words",
            "false": "at least one of destination, dates or traveller count was not said by the "
                     "user and was assumed, guessed or invented by the assistant",
        },
    ),
    "explicit_request": Decider.noul(
        "Is the user's most recent message a direct instruction to complete the booking now, "
        "rather than a question about options, prices or availability?",
        {
            "true": "an instruction such as 'book it', 'go ahead', 'reserve that flight', "
                    "'book us two rooms', or naming a specific flight/hotel to confirm",
            "false": "a question about options, prices, or availability, a search request, "
                     "or a statement that does not request confirmation",
        },
    ),
}
```

Search tools only need search_grounded (one question, one forward pass). Booking tools need
args_grounded and explicit_request (two questions, one forward pass -- all questions about
one state share one encoding). If the decider returns guide, TravelBookingGuard adds the call
signature to a _guided set. If the agent retries the identical call unchanged, it gets Deny
instead of Guide (Strands requires guide handlers to converge).

< FIGURE 2 of 5: upload submission/article/img/architecture.png here >
Alt text: Architecture diagram showing the user, the Strands Agents agent loop with tools and TravelBookingGuard, Amazon Bedrock for the LLM, and Strands Decider 2B at the intervention seam.

## Measured

The benchmark ran on the test split of eval_cases.jsonl against Amazon Nova Pro as the LLM
baseline. Both were asked the same typed questions about the same 12 labelled cases.

| | Strands Decider 2B | Amazon Nova Pro |
|---|---|---|
| Accuracy | 79.2% | 95.8% |
| Latency p50 per decision | 295 ms | 1276 ms |
| Latency p95 | 468 ms | 1426 ms |
| Cost per 1,000 decisions | $0.00 | $0.18 |
| Confident answers (>= 0.70) | 67% | 100% |
| Accuracy when confident | 93.8% | 95.8% |

At the median the decider was 4.3x faster and free. When it was confident (67% of decisions)
its accuracy matched the LLM at 93.8%. The 21% of decisions where the decider was wrong or
unsure either went to the human-confirm path or were outweighed by the latency and cost savings
on the 67% where it was right and fast.

The weak spot is explicit_request: the decider was right on 66.7% of those decisions versus
100% for the LLM. That question distinguishes "book it" from "how much is it?" which can be
genuinely subtle. Tighter criteria descriptions in QUESTIONS would improve this; the kit's
tune/test split keeps the tuning honest.

< FIGURE 3 of 5: upload submission/article/img/bench-accuracy.png here >
Alt text: Bar chart showing Strands Decider 2B at 79.2% accuracy and Amazon Nova Pro at 95.8% on the labelled eval cases.

< FIGURE 4 of 5: upload submission/article/img/bench-latency.png here >
Alt text: Bar chart showing median latency per decision: 295 ms for Strands Decider 2B, 1276 ms for Amazon Nova Pro.

< FIGURE 5 of 5: upload submission/article/img/bench-confidence.png here >
Alt text: Chart showing accuracy vs. share of decisions answered without a human as the confidence threshold moves from 0 to 1.

## The reusable decision pattern

The entire guardrail lives in agent/decision.py. The docstring is tagged
"REUSABLE DECISION PATTERN: travel-booking argument-grounding guardrail (#guardrails)".
To drop it into a different agent:

1. Copy agent/decision.py.
2. Set BOOKING_TOOLS to the irreversible tools in your domain.
3. Set SEARCH_TOOLS to the query tools you want to guard (or remove that path entirely if
   searches in your domain carry no arguments worth guarding).
4. Reword the typed questions for the arguments your domain requires.
5. Wire TravelBookingGuard into your Strands Agent:
   Agent(..., interventions=[TravelBookingGuard(Decider())])

The convergence guard (_guided set + Deny on retry) and the unsure-band Confirm path both
carry over unchanged. The five threshold and band parameters (yes, unsure_band) are tunable
per deployment.

## What I learned and what is next

The decider is not a replacement for the LLM on reasoning tasks -- it is a fast, cheap, and
calibrated gate for decisions that have a clear yes/no shape. Argument grounding is that shape
exactly: either the user said the city or they did not. The 4.3x speedup and zero cost make
the check viable before every single tool call, which a full LLM round-trip would not be.

What I would improve: tighten the explicit_request question criteria to separate soft
confirmations ("that sounds good") from hard ones ("book it"), add a fourth question for
traveller-identity grounding (was a passenger name given?), and run the benchmark on the full
16-case set including the new search-tool cases.

---

Code: TODO repo link . Video: TODO video link . Tags: #strands-decider-2b #guardrails

Built with the AWS User Group Madurai Strands Decider hackathon kit
