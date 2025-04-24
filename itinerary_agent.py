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

# Rendszerprompt az útiterv tervezéshez
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
    
    # 4. lépés: Végső útiterv generálása az LLM-mel
    prompt = f"""
    Készíts egy budapesti útitervet a következő részletek alapján:
    
    Kiindulási hely: {start_location}
    Érdeklődési körök: {', '.join(interests)}
    Rendelkezésre álló idő: {available_time} óra
    Közlekedési mód: {transport_mode}
    Speciális kérések: {special_requests}
    
    Kiválasztott látványosságok:
    {json.dumps(selected_attractions)}
    
    Látványosság információk (webes keresésből):
    {attraction_descriptions}
    
    Formázd az útitervet a következőkkel:
    1. Egy cím és rövid bevezető
    2. Egy időbeosztás 10:00-tól kezdve
    3. Részletek minden látványosságról, beleértve:
       - Leírás (használd a pontos információkat a webes keresésből)
       - Szükséges idő a látogatáshoz
       - Közlekedési utasítások
    4. Étkezési javaslatok megfelelő időpontokban
    
    Legyen az útiterv vizuálisan rendezett és könnyen követhető.
    """
    
    messages = [
        SystemMessage(content=ITINERARY_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = planning_llm.invoke(messages)
    return response.content

def get_attraction_descriptions_with_search(attractions):
    """Pontos leírások lekérése látványosságokhoz webes keresési képesség használatával"""
    prompt = f"""
    Webes kereséshez hozzáféréssel rendelkezel, hogy pontos információkat nyújts budapesti látványosságokról.
    
    A következő budapesti látványosságok mindegyikéről adj rövid, de részletes leírást aktuális webes információk alapján:
    {json.dumps(attractions)}
    
    Minden látványosságnál szerepeljenek a következők:
    1. Mi ez (múzeum, nevezetesség stb.)
    2. Történelmi jelentősége
    3. Főbb jellemzői és amit a látogatók láthatnak
    4. Elhelyezkedése Budapesten
    5. Praktikus látogatói információk (ha elérhetők)
    
    Formázd minden leírást a látványosság nevével fejlécként, majd 3-4 informatív mondattal.
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
