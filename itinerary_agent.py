# itinerary_agent.py
# Egyszerű útiterv tervező a Budapest Explorer alkalmazáshoz
# Szerző: Szalay Miklós Márton
# Szakdolgozat projekt a Pannon Egyetem számára

from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Importáljuk a nyers függvényeket az agent.py-ból az eszköz wrapperek helyett
from agent import (
    OPENAI_API_KEY,
    parse_trip_input,
    get_directions,
    get_local_attractions,
    extract_attraction_names
)

# LLM-ek inicializálása - normál a tervezéshez és keresés-támogatott a látványosság információkhoz
planning_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)
search_llm = ChatOpenAI(model="gpt-4o-search-preview-2025-03-11", openai_api_key=OPENAI_API_KEY)

# System prompt for itinerary planning
ITINERARY_PROMPT = """
You are a Budapest itinerary planning assistant. Your task is to create personalized itineraries based on the provided information.

Create a detailed itinerary that includes:
- A list of attractions based on the user's interests
- A time schedule for each location
- Routes between locations
- Meal suggestions
- A brief description of each location

Always respond in the same language as the user's request.
"""

def create_itinerary(preferences):
    """Útiterv létrehozása felhasználói preferenciák alapján"""
    # Kiindulási hely lekérése
    start_location = preferences.get("start_location", "Deák Ferenc tér")
    interests = preferences.get("interests", [])
    available_time = preferences.get("available_time", 4)
    transport_mode = preferences.get("transport_mode", "transit")
    special_requests = preferences.get("special_requests", "")
    
    # 1. lépés: Látványosságok keresése érdeklődési körök alapján
    attractions = []
    
    # Látványosságnevek kinyerése a speciális kérésekből, ha vannak
    if special_requests:
        # Közvetlenül használjuk a nyers függvényt
        extracted_attractions = extract_attraction_names(special_requests)
        attractions.extend(extracted_attractions)
    
    # Ha az érdeklődési körök tartalmaznak konkrét kategóriákat, keressünk több látványosságot
    if not attractions or len(attractions) < 3:
        # Közvetlenül használjuk a get_directions függvényt
        route_data = get_directions(
            from_place=start_location,
            to_place="Hősök tere, Budapest",
            mode=transport_mode
        )
        
        # Koordináták kinyerése az útvonalból
        if "routes" in route_data and route_data["routes"]:
            leg = route_data["routes"][0]["legs"][0]
            lat = leg["start_location"]["lat"]
            lng = leg["start_location"]["lng"]
            
            # Látványosságok lekérése minden érdeklődési kategóriához
            for interest in interests:
                category = map_interest_to_category(interest)
                # Közvetlenül használjuk a get_local_attractions függvényt
                attractions_result = get_local_attractions(
                    lat=lat,
                    lng=lng,
                    category=category,
                    radius=1000
                )
                
                if "places" in attractions_result:
                    for place in attractions_result["places"]:
                        attractions.append(place["name"])
    
    # Korlátozás a legjobb látványosságokra a rendelkezésre álló idő alapján
    max_attractions = min(int(available_time) // 2 + 1, 5)
    selected_attractions = attractions[:max_attractions]
    
    # Ha nem találtunk látványosságokat, adjunk hozzá néhány alapértelmezett látványosságot
    if not selected_attractions:
        selected_attractions = ["Parlament", "Budai vár", "Halászbástya"]
    
    # 2. lépés: Látványosság információk lekérése a keresés-támogatott modellel
    attraction_descriptions = get_attraction_descriptions_with_search(selected_attractions)
    
    # 3. lépés: Útvonalak tervezése a látványosságok között
    routes = []
    current_location = start_location
    
    for attraction in selected_attractions:
        # Közvetlenül használjuk a get_directions függvényt
        route = get_directions(
            from_place=current_location,
            to_place=attraction + ", Budapest",
            mode=transport_mode
        )
        routes.append(route)
        current_location = attraction + ", Budapest"
    
    # 4. Step: Generate the final itinerary with the LLM
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
    
    Respond in the same language as the user's query.
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
    
    Respond in the same language as the user's query.
    """
    
    response = search_llm.invoke([HumanMessage(content=prompt)])
    return response.content

def map_interest_to_category(interest):
    """Felhasználói érdeklődési körök leképezése Google Places API kategóriákra"""
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
