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

# Import the raw functions from agent.py instead of the tool wrappers
from agent import (
    OPENAI_API_KEY,
    parse_trip_input,
    get_directions,
    get_local_attractions,
    extract_attraction_names
)

# Initialize the LLMs - regular for planning and search-enabled for attraction info
planning_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)
search_llm = ChatOpenAI(model="gpt-4o-search-preview-2025-03-11", openai_api_key=OPENAI_API_KEY, temperature=0.2)

# System prompt for itinerary planning
ITINERARY_PROMPT = """
Te egy Budapest útiterv-készítő asszisztens vagy. Feladatod személyre szabott útvonaltervet készíteni a megadott információk alapján.

Készíts részletes útitervet, amely tartalmazza:
- A látványosságok listáját a felhasználó érdeklődése alapján
- Időtervet az egyes helyszínekre
- Útvonalakat a helyszínek között
- Étkezési javaslatokat
- Rövid leírást minden helyszínről
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
        # Use the raw function directly
        extracted_attractions = extract_attraction_names(special_requests)
        attractions.extend(extracted_attractions)
    
    # If interests include specific categories, find more attractions
    if not attractions or len(attractions) < 3:
        # Use get_directions function directly
        route_data = get_directions(
            from_place=start_location,
            to_place="Hősök tere, Budapest",
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
                # Use get_local_attractions function directly
                attractions_result = get_local_attractions(
                    lat=lat,
                    lng=lng,
                    category=category,
                    radius=1000
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
    
    # Step 2: Get attraction information using the search-enabled model
    attraction_descriptions = get_attraction_descriptions_with_search(selected_attractions)
    
    # Step 3: Plan routes between attractions
    routes = []
    current_location = start_location
    
    for attraction in selected_attractions:
        # Use get_directions function directly  
        route = get_directions(
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
    
    Attraction information (from web search):
    {attraction_descriptions}
    
    Format the itinerary with:
    1. A title and brief introduction
    2. A time schedule starting at 10:00 AM
    3. Details for each attraction including:
       - Description (use the accurate information from web search)
       - Time needed to visit
       - Transportation instructions
    4. Meal suggestions at appropriate times
    
    Make the itinerary visually organized and easy to follow.
    """
    
    messages = [
        SystemMessage(content=ITINERARY_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = planning_llm.invoke(messages)
    return response.content

def get_attraction_descriptions_with_search(attractions):
    """Get accurate descriptions for attractions using web search capability"""
    prompt = f"""
    You have access to web search to provide accurate information about Budapest attractions.
    
    For each of these Budapest attractions, provide a brief but detailed description based on current web information:
    {json.dumps(attractions)}
    
    For each attraction, include:
    1. What it is (museum, landmark, etc.)
    2. Historical significance
    3. Key features and what visitors can see
    4. Location in Budapest
    5. Any practical visitor information (if available)
    
    Format each description with the attraction name as a header followed by 3-4 informative sentences.
    """
    
    response = search_llm.invoke([HumanMessage(content=prompt)])
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
