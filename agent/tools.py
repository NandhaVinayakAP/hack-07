"""Travel-booking agent tools: flights and hotels, with synthetic in-memory data.

All data is synthetic so the demo runs anywhere with no external API keys.
Replace with real airline / hotel API calls for production.
Keep the one-line docstring on each tool: the model reads it.
"""

from __future__ import annotations

from strands import tool

# ---------------------------------------------------------------------------
# Synthetic flight data
# ---------------------------------------------------------------------------
_FLIGHTS: list[dict] = [
    {"flight_id": "FL001", "origin": "New York", "destination": "London",
     "date": "2025-11-01", "seats": 12, "price_usd": 620},
    {"flight_id": "FL002", "origin": "New York", "destination": "London",
     "date": "2025-11-01", "seats": 5, "price_usd": 780},
    {"flight_id": "FL003", "origin": "London", "destination": "Paris",
     "date": "2025-11-03", "seats": 20, "price_usd": 95},
    {"flight_id": "FL004", "origin": "New York", "destination": "Paris",
     "date": "2025-11-02", "seats": 8, "price_usd": 710},
    {"flight_id": "FL005", "origin": "San Francisco", "destination": "Tokyo",
     "date": "2025-11-05", "seats": 6, "price_usd": 950},
]

_BOOKINGS: list[dict] = []

# ---------------------------------------------------------------------------
# Synthetic hotel data
# ---------------------------------------------------------------------------
_HOTELS: list[dict] = [
    {"hotel_id": "HT001", "city": "London", "name": "Grand Kensington",
     "stars": 4, "price_per_night_usd": 180, "rooms_available": 10},
    {"hotel_id": "HT002", "city": "London", "name": "Budget Inn Shoreditch",
     "stars": 2, "price_per_night_usd": 75, "rooms_available": 3},
    {"hotel_id": "HT003", "city": "Paris", "name": "Hotel Lumière",
     "stars": 4, "price_per_night_usd": 210, "rooms_available": 7},
    {"hotel_id": "HT004", "city": "Paris", "name": "Pension du Marais",
     "stars": 3, "price_per_night_usd": 120, "rooms_available": 5},
    {"hotel_id": "HT005", "city": "Tokyo", "name": "Shinjuku Grand",
     "stars": 5, "price_per_night_usd": 320, "rooms_available": 4},
]

_HOTEL_BOOKINGS: list[dict] = []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_flights(origin: str, destination: str, date: str, travellers: int) -> dict:
    """Search available flights from origin to destination on a date for a number of travellers."""
    results = [
        f for f in _FLIGHTS
        if f["origin"].lower() == origin.lower()
        and f["destination"].lower() == destination.lower()
        and f["date"] == date
        and f["seats"] >= travellers
    ]
    if not results:
        return {"flights": [], "message": f"No flights found from {origin} to {destination} on {date} for {travellers} traveller(s)."}
    return {"flights": results, "count": len(results)}


@tool
def book_flight(flight_id: str, travellers: int, passenger_name: str) -> dict:
    """Book a flight by its flight id for a number of travellers under a passenger name (irreversible)."""
    flight = next((f for f in _FLIGHTS if f["flight_id"] == flight_id), None)
    if not flight:
        return {"error": f"No flight with id {flight_id}."}
    if flight["seats"] < travellers:
        return {"error": f"Only {flight['seats']} seat(s) available, requested {travellers}."}
    flight["seats"] -= travellers
    booking = {
        "booking_ref": f"BK{len(_BOOKINGS) + 1001}",
        "flight_id": flight_id,
        "origin": flight["origin"],
        "destination": flight["destination"],
        "date": flight["date"],
        "travellers": travellers,
        "passenger_name": passenger_name,
        "total_usd": flight["price_usd"] * travellers,
    }
    _BOOKINGS.append(booking)
    return booking


@tool
def search_hotels(city: str, check_in: str, check_out: str, guests: int) -> dict:
    """Search available hotels in a city for a date range and number of guests."""
    results = [h for h in _HOTELS if h["city"].lower() == city.lower() and h["rooms_available"] >= 1]
    if not results:
        return {"hotels": [], "message": f"No hotels found in {city} for {guests} guest(s) from {check_in} to {check_out}."}
    # Include the date range and guest count in the response for the LLM's context.
    return {"hotels": results, "city": city, "check_in": check_in, "check_out": check_out, "guests": guests}


@tool
def book_hotel(hotel_id: str, check_in: str, check_out: str, guests: int, guest_name: str) -> dict:
    """Book a hotel room by its hotel id for a date range, number of guests and guest name (irreversible)."""
    hotel = next((h for h in _HOTELS if h["hotel_id"] == hotel_id), None)
    if not hotel:
        return {"error": f"No hotel with id {hotel_id}."}
    if hotel["rooms_available"] < 1:
        return {"error": f"No rooms available at {hotel['name']}."}
    # Compute nights from dates (simple subtraction via datetime).
    from datetime import date as _date
    try:
        nights = (_date.fromisoformat(check_out) - _date.fromisoformat(check_in)).days
    except ValueError:
        return {"error": "Invalid date format; use YYYY-MM-DD."}
    if nights <= 0:
        return {"error": "Check-out must be after check-in."}
    hotel["rooms_available"] -= 1
    booking = {
        "booking_ref": f"HB{len(_HOTEL_BOOKINGS) + 2001}",
        "hotel_id": hotel_id,
        "hotel_name": hotel["name"],
        "city": hotel["city"],
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "guests": guests,
        "guest_name": guest_name,
        "total_usd": hotel["price_per_night_usd"] * nights,
    }
    _HOTEL_BOOKINGS.append(booking)
    return booking


TOOLS = [search_flights, book_flight, search_hotels, book_hotel]
