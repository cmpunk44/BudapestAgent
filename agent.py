# agent.py
# LangGraph-alapú ágens budapesti turizmus és közlekedési információkhoz
# Szerző: Szalay Miklós Márton
# Szakdolgozat projekt a Pannon Egyetem számára

from dotenv import load_dotenv
load_dotenv()  # Környezeti változók betöltése a .env fájlból

import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated, List, Dict, Any

# LangChain komponensek importálása
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# API kulcsok betöltése környezeti változókból
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# LLM inicializálása OpenAI-val
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# === Eszköz függvények ===

def parse_trip_input(user_input: str) -> dict:
    """Extract origin and destination from user text input."""
    prompt = f"""
    You are a multilingual assistant specializing in Hungarian location recognition.
    Extract two locations from this sentence.
    Be flexible with Hungarian address formats and landmarks in Budapest.
    Respond ONLY with a JSON like:
    {{"from": "X", "to": "Y"}}
    Input: "{user_input}"
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    try:
        return json.loads(response.content)
    except:
        # Default empty response if parsing fails
        return {"from": "", "to": ""}


def get_directions(from_place: str, to_place: str, mode: str = "transit") -> dict:
    """Útvonal keresése a Google Directions API segítségével."""
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Budapest hozzáadása a helyszínhez, ha nincs megadva
    if "budapest" not in from_place.lower():
        from_place += ", Budapest, Hungary"
    if "budapest" not in to_place.lower():
        to_place += ", Budapest, Hungary"
        
    params = {
        "origin": from_place,
        "destination": to_place,
        "mode": mode,
        "language": "hu",
        "key": MAPS_API_KEY
    }
    
    # Tömegközlekedés-specifikus paraméterek hozzáadása
    if mode == "transit":
        params["transit_mode"] = "bus|subway|train|tram"
    
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {"error": "Directions API sikertelen"}

def get_local_attractions(lat: float, lng: float, category: str = "tourist_attraction", radius: int = 1000) -> dict:
    """Helyek keresése koordináták és kategória alapján a Google Places API segítségével."""
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    # Felhasználóbarát kategóriák leképezése Google Places API típusokra
    category_map = {
        "attractions": "tourist_attraction",
        "restaurants": "restaurant",
        "cafes": "cafe",
        "museums": "museum",
        "parks": "park",
        "shopping": "shopping_mall",
        "art_gallery": "art_gallery",
        "church": "church",
        "historical_site": "establishment",
        "theater": "movie_theater",
        "hotel": "lodging",
        "nightlife": "night_club",
        "spa": "spa",
        "bakery": "bakery",
        "viewpoint": "point_of_interest",
        "entertainment": "amusement_park",
    }
    
    place_type = category_map.get(category.lower(), category)
    
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": place_type,
        "language": "hu",
        "key": MAPS_API_KEY
    }
    
    res = requests.get(places_url, params=params)
    if res.status_code == 200:
        data = res.json()
        places = []
        for place in data.get("results", [])[:5]:  # Legfeljebb 5 eredményre korlátozzuk
            places.append({
                "name": place.get("name"),
                "rating": place.get("rating", "N/A"),
                "address": place.get("vicinity"),
                "open_now": place.get("opening_hours", {}).get("open_now", "unknown")
            })
        return {"places": places}
    return {"error": "Places API sikertelen", "places": []}

def extract_attraction_names(text: str) -> list:
    """Extract attraction names from user query text."""
    prompt = f"""
    You are a specialized assistant for Budapest tourism.
    From the following text, extract any mentioned or implied Budapest attractions, landmarks, or places of interest.
    Return ONLY a JSON array of attraction names, with no additional text.
    
    Example output: ["Parliament Building", "Buda Castle"]
    
    Text: "{text}"
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    try:
        attractions = json.loads(response.content)
        if isinstance(attractions, list):
            return attractions
    except:
        # Potenciális tulajdonnevek keresése (nagybetűvel kezdődő szavak)
        potential_attractions = re.findall(r'([A-Z][a-zA-Záéíóöőúüű]+(?:\s+[A-Z][a-zA-Záéíóöőúüű]+)*)', text)
        return potential_attractions[:3] if potential_attractions else []

# === Eszközök regisztrálása a LangChain @tool dekorátorral ===

@tool
def parse_input_tool(text: str) -> dict:
    """Felhasználói bemenet feldolgozása és 'from' és 'to' úticélok kinyerése."""
    return parse_trip_input(text)

@tool
def directions_tool(from_place: str, to_place: str, mode: str = "transit") -> dict:
    """Útvonal lekérése a Google Directions API segítségével.
    Args:
        from_place: Kiindulási hely
        to_place: Célállomás
        mode: Közlekedési mód (transit, walking, bicycling, driving)
    """
    return get_directions(from_place, to_place, mode)

@tool
def attractions_tool(lat: float, lng: float, category: str = "tourist_attraction", radius: int = 1000) -> dict:
    """Helyek keresése koordináták és kategória alapján.
    Args:
        lat: Szélesség
        lng: Hosszúság
        category: Hely kategória (attractions, restaurants, cafes, museums, parks, shopping)
        radius: Keresési sugár méterben
    """
    return get_local_attractions(lat, lng, category, radius)

@tool
def extract_attractions_tool(text: str) -> list:
    """Látványosságok neveinek kinyerése a felhasználó kérdéséből.
    Args:
        text: A felhasználó kérdésének szövege
    """
    return extract_attraction_names(text)

@tool
def attraction_info_tool(attractions: list) -> dict:
    """
    Provides information about Budapest attractions using web search.
    Args:
        attractions: A list of attraction names to get information about
    """
    if not attractions or len(attractions) == 0:
        return {"info": "No attractions specified.", "source": "web search"}
    
    prompt = f"""
You are a tourist assistant specialized in Budapest.
Please provide a short (max 3 sentences) Budapest-specific description for each of the following tourist attractions:
{json.dumps(attractions, indent=2)}
Focus ONLY on Budapest context. No global or irrelevant content.
Return a list where each name is followed by its description.
"""
    try:
        # Keresőképességgel rendelkező modell használata
        gpt4_model = ChatOpenAI(model="gpt-4o-search-preview-2025-03-11", openai_api_key=OPENAI_API_KEY)
        response = gpt4_model.invoke([HumanMessage(content=prompt)])
        
        return {
            "info": response.content,
            "source": "web keresés",
            "attractions": attractions
        }
    except Exception as e:
        return {
            "info": f"Hiba az információk lekérése során: {str(e)}",
            "source": "hiba",
            "attractions": attractions
        }

# === Ágens állapot definiálása ===
class AgentState(TypedDict):
    """Az ágens állapotát reprezentálja a beszélgetés során."""
    messages: Annotated[list[AnyMessage], operator.add]  # Az üzenetek halmozódnak

# === Ágens osztály a beszélgetési folyamat kezeléséhez ===
class Agent:
    """LangGraph-alapú ágens, amely eszközöket használhat budapesti turisztikai kérdések megválaszolásához."""
    
    def __init__(self, model, tools, system=""):
        """Az ágens inicializálása nyelvi modellel, eszközökkel és rendszerprompttal."""
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        # Egyszerű gráf létrehozása két csomóponttal: LLM és action
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)  # Csomópont válaszok vagy eszközhívások generálásához
        graph.add_node("action", self.take_action)  # Csomópont eszközök végrehajtásához
        
        # Élek hozzáadása a folyamat definiálásához
        graph.add_conditional_edges(
            "llm",  # Az LLM csomópontból
            self.exists_action,  # Ellenőrzi, hogy van-e hívandó eszköz
            {True: "action", False: END}  # Ha igen, action-höz megy; ha nem, befejezi
        )
        graph.add_edge("action", "llm")  # Action után vissza az LLM-hez
        
        # Belépési pont beállítása
        graph.set_entry_point("llm")
        
        # A gráf fordítása
        self.graph = graph.compile()

    def exists_action(self, state: AgentState):
        """Ellenőrzi, hogy az utolsó üzenet tartalmaz-e eszközhívásokat."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(getattr(result, 'tool_calls', [])) > 0

    def call_openai(self, state: AgentState):
        """Meghívja a nyelvi modellt válasz vagy eszközhívások generálásához."""
        messages = state['messages']
        
        # Rendszerüzenet hozzáadása, ha nincs jelen
        if self.system and not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=self.system)] + messages
            
        # A modell meghívása és válasz kérése
        message = self.model.invoke(messages)
        
        # Frissített állapot visszaadása az új üzenettel
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        """Végrehajtja a nyelvi modell által kért eszközhívásokat."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        # Minden eszközhívás feldolgozása
        for t in tool_calls:
            if t['name'] not in self.tools:
                result = f"Érvénytelen eszköznév: {t['name']}. Próbáld újra."
            else:
                try:
                    # Az eszköz meghívása az argumentumokkal
                    result = self.tools[t['name']].invoke(t['args'])
                except Exception as e:
                    result = f"Hiba az eszköz végrehajtása során: {str(e)}"
                    
            # ToolMessage létrehozása az eredménnyel
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            
        # Frissített állapot visszaadása az eszköz eredményeivel
        return {'messages': results}

# === System prompt for the agent ===
prompt = """
You are a helpful assistant for Budapest public transport and sightseeing.
You help tourists and locals navigate Budapest and discover interesting places.

Follow these steps when responding to users:

1. For route planning:
   - Extract origin and destination from user input using parse_input_tool
   - Call directions_tool with both locations to get a route
   - Format the route results into a user-friendly summary following these guidelines:
     * Start with a header showing origin → destination
     * Include the total duration and distance
     * List each step of the journey with appropriate icons:
       - 🚆 for transit vehicles (showing line numbers, vehicle types, and stop names)
       - 🚶 for walking segments (showing duration)
     * Format step numbers and use clear arrows (→) between locations
     * For transit steps, include: line number, vehicle type, departure stop, and arrival stop
   - If the user specifies a transportation mode (walking, bicycling, driving), use that mode

2. For attraction recommendations:
   - Get coordinates from the route data
   - Call attractions_tool with relevant coordinates
   - If the user specifies a category (restaurants, cafes, etc.), use that category
   - After getting attractions, use attraction_info_tool to get accurate descriptions

3. For specific information about attractions:
   - Use extract_attractions_tool first to identify attraction names in the query
   - Then use attraction_info_tool with those attraction names
   - When showing attraction information, CLEARLY mention you got this from web search

IMPORTANT RULES:
- When users ask about attractions in Budapest, always use the web search capability
- First extract attraction names with extract_attractions_tool, then look them up with attraction_info_tool
- Explicitly state that information comes from "web search" in your responses
- When formatting route information from directions_tool, pay special attention to:
  * Use a consistent, readable format with proper spacing and organization
  * For transit routes, clearly indicate line numbers and vehicle types (bus, tram, metro)
  * Include emojis to represent different transportation modes (🚆, 🚍, 🚇, 🚶)
  * Format durations and distances in a readable way
  * Structure step-by-step directions with clear numbering
  * Handle error cases gracefully (route not found, invalid locations)
- Always end your responses with 1-2 relevant follow-up questions based on the information provided:
  * For route planning: Ask about attractions or restaurants near the destination
  * For attraction information: Ask if they want to know about nearby places or how to get there
  * For general queries: Suggest related topics or activities in Budapest

Always respond in the same language the user used to ask their question.
Be helpful, friendly, and provide concise but complete information.
"""

# Modell példány létrehozása
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Az ágenshez elérhető eszközök definiálása
tools = [
    parse_input_tool, 
    directions_tool, 
    attractions_tool,
    extract_attractions_tool,
    attraction_info_tool
]

# Ágens példány létrehozása
budapest_agent = Agent(model, tools, system=prompt)
