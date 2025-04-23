# itinerary_agent.py
# Simple itinerary planner for Budapest Explorer
# Author: Szalay Miklós Márton
# Thesis project for Pannon University

from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import tools from the main agent
from agent import (
    parse_input_tool,
    directions_tool,
    attractions_tool,
    extract_attractions_tool,
    attraction_info_tool,
    OPENAI_API_KEY
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# System prompt for itinerary planning
ITINERARY_PROMPT = """
Te egy Budapest útiterv-készítő asszisztens vagy. Feladatod személyre szabott útvonaltervet készíteni a megadott információk alapján.

Készíts részletes útitervet, amely tartalmazza:
- A látványosságok listáját a felhasználó érdeklődése alapján
- Időtervet az egyes helyszínekre
- Útvonalakat a helyszínek között
- Étkezési javaslatokat
- Rövid leírást minden helyszínről

You are a Budapest itinerary planner assistant. Your task is to create a personalized itinerary based on the provided information.

Create a detailed itinerary that includes:
- List of attractions based on user interests
- Time schedule for each location
- Routes between locations
- Meal suggestions
- Brief description of each location
"""

def create_itinerary(preferences):
    """Create an itinerary based on user preferences"""
    # Get starting location
    start_location = preferences.get("start_location", "Deák Ferenc tér")
    interests = preferences.get("interests", [])
    available_time = preferences.get("available_time", 4)
    transport_mode = preferences.get("transport_mode", "transit")
    special_requests = preferences.get("special_requests", "")
    
    # Step 1: Find attractions based on interests
    attractions = []
    
    # Extract attraction names from special requests if any
    if special_requests:
        # The tool requires 'text' as the parameter name
        extracted_attractions = extract_attractions_tool(text=special_requests)
        attractions.extend(extracted_attractions)
    
    # If interests include specific categories, find more attractions
    if not attractions or len(attractions) < 3:
        # Use directions_tool to get coordinates from starting location
        route_data = directions_tool(
            from_place=start_location,
            to_place="Hősök tere, Budapest",  # Common tourist destination
            mode=transport_mode
        )
        
        # Extract coordinates from the route
        if "routes" in route_data and route_data["routes"]:
            leg = route_data["routes"][0]["legs"][0]
            lat = leg["start_location"]["lat"]
            lng = leg["start_location"]["lng"]
            
            # Get attractions for each interest category
            for interest in interests:
                category = map_interest_to_category(interest)
                attractions_result = attractions_tool(
                    lat=lat,
                    lng=lng,
                    category=category,
                    radius=3000
                )
                
                if "places" in attractions_result:
                    for place in attractions_result["places"]:
                        attractions.append(place["name"])
    
    # Limit to top attractions based on available time
    max_attractions = min(int(available_time) // 2 + 1, 5)
    selected_attractions = attractions[:max_attractions]
    
    # If no attractions were found, add some default attractions
    if not selected_attractions:
        selected_attractions = ["Parliament", "Buda Castle", "Fisherman's Bastion"]
    
    # Step 2: Get attraction information
    attraction_info = attraction_info_tool(attractions=selected_attractions)
    
    # Step 3: Plan routes between attractions
    routes = []
    current_location = start_location
    
    for attraction in selected_attractions:
        route = directions_tool(
            from_place=current_location,
            to_place=attraction + ", Budapest",
            mode=transport_mode
        )
        routes.append(route)
        current_location = attraction + ", Budapest"
    
    # Step 4: Generate the final itinerary with the LLM
    prompt = f"""
    Create a Budapest itinerary based on these details:
    
    Starting location: {start_location}
    Interests: {', '.join(interests)}
    Available time: {available_time} hours
    Transportation mode: {transport_mode}
    Special requests: {special_requests}
    
    Selected attractions:
    {json.dumps(selected_attractions)}
    
    Attraction information:
    {attraction_info.get('info', 'Information not available')}
    
    Format the itinerary with:
    1. A title and brief introduction
    2. A time schedule starting at 10:00 AM
    3. Details for each attraction including:
       - Description
       - Time needed to visit
       - Transportation instructions
    4. Meal suggestions at appropriate times
    
    Make the itinerary visually organized and easy to follow.
    """
    
    messages = [
        SystemMessage(content=ITINERARY_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content

def map_interest_to_category(interest):
    """Map user interests to Google Places API categories"""
    interest_map = {
        "museums": "museum",
        "history": "tourist_attraction",
        "architecture": "tourist_attraction",
        "food": "restaurant",
        "nature": "park",
        "shopping": "shopping_mall",
        "art": "art_gallery",
        "nightlife": "night_club",
        "culture": "tourist_attraction",
        "religion": "church"
    }
    
    return interest_map.get(interest.lower(), "tourist_attraction")
