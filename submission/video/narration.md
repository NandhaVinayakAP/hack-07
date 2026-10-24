0. This travel-booking agent talks through Amazon Bedrock. For this demo it is told to act immediately without asking, and Strands Decider checks every tool call before it runs — both searches and bookings.

1. A travel-booking agent that searches flights and hotels and cannot run any tool — search or booking — unless every argument (city, dates, traveller count) was explicitly stated by the user. Strands Decider checks every tool call before it runs. What it does: Search and book flights and hotels through chat. Check every tool call — search and booking — with Strands Decider 2B before it runs. Ask a follow-up question instead of guessing missing cities, dates, or traveller counts. Deployed on Amazon Bedrock AgentCore Runtime.

2. The user mentions a destination but gives no origin, date, or traveller count. The agent tries to call search_flights with invented values. The decider scored search grounded 0.32, so the verdict was guide, in 4.2 seconds.

3. Now the user provides origin, date, and traveller count. The search arguments are fully grounded. The decider scored search grounded 0.84, so the verdict was proceed, in 3.8 seconds.

4. The user explicitly names the flight and asks to book it. The booking arguments are grounded and the request is explicit. The decider scored args grounded 0.80, explicit request 0.82, so the verdict was proceed, in 5.5 seconds.

5. The user asks for a hotel search but gives no check-in or check-out dates. The agent tries to call search_hotels with invented dates. The decider scored search grounded 0.46, so the verdict was guide, in 5.1 seconds.

6. The user provides dates and names the hotel. Both the search and the booking are now fully grounded. The decider scored args grounded 0.72, explicit request 0.83, so the verdict was proceed, in 6.7 seconds.

7. On 24 labelled decisions, with the decider running locally on this machine, Strands Decider was 79% accurate at a median of 295 milliseconds, against 96% and 1276 milliseconds for the Bedrock LLM asked the same questions.

8. One forward pass per tool-call decision, calibrated confidence, and Amazon Bedrock for the conversation. Every search and booking is checked before it runs.