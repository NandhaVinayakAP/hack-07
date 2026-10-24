# Use case

- **Pitch:** a travel-booking agent that searches and books flights and hotels, and cannot
  search or book anything unless every key argument — origin/destination city, dates, and
  number of travellers — was actually stated by the user, not invented by the LLM.
- **Who it is for:** travellers who want to book a trip through a chat interface, and the
  travel platform that wants confident, grounded searches and bookings without human review
  of every request.
- **Features:**
  - flight search and booking, hotel search and booking through chat;
  - **every tool call** (search and booking) is checked before it runs: if any argument
    (city, check-in/check-out date, traveller count) is not grounded in the conversation,
    the agent asks a follow-up question instead of guessing;
  - search tools are guarded by a `search_grounded` question — the agent cannot fire
    `search_flights` or `search_hotels` with invented origins, cities or dates;
  - booking tools receive two additional checks: `args_grounded` (were all booking arguments
    stated?) and `explicit_request` (is the user actually asking to book right now?);
  - a live decision panel in the demo UI that shows why each tool call was allowed or held back;
  - deployed on Amazon Bedrock AgentCore Runtime.
- **The decisions handed to Strands Decider 2B:** (`#guardrails`)
  - before any search: are the search arguments (city, dates, traveller count) grounded in
    what the user said? (`search_grounded`)
  - before any booking: are all booking arguments grounded? (`args_grounded`) and is the
    user asking to book right now? (`explicit_request`)
- **Why a decision model:** the grounding check runs before every tool call; it must be fast,
  cheap and consistent, and it must return a number the code can branch on without a second
  round-trip to the LLM.
- **Amazon Bedrock:** the conversation and tool use (Amazon Nova Pro by default).
