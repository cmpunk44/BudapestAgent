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
    """Kiindulási és célállomás kinyerése a felhasználói szövegből."""
    prompt = f"""
    Te egy többnyelvű asszisztens vagy, aki a magyar helyszínek felismerésére specializálódott.
    Vonj ki két helyszínt ebből a mondatból.
    Légy rugalmas a magyar címformátumokkal és budapesti nevezetességekkel.
    Válaszolj KIZÁRÓLAG JSON formátumban, így:
    {{"from": "X", "to": "Y"}}
    Bemenet: "{user_input}"
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    try:
        return json.loads(response.content)
    except:
        # Egyszerű regex alapú feldolgozás
        match = re.search(r'from\s+(.*?)\s+to\s+(.*)', user_input, re.IGNORECASE)
        if match:
            return {"from": match.group(1), "to": match.group(2)}
        # Magyar mintázatok próbálása
        match = re.search(r'(.*?)-(?:ról|ről|ból|ből|tól|től)\s+(?:a |az )?(.*?)(?:-ra|-re|-ba|-be|-hoz|-hez|-höz)?', user_input, re.IGNORECASE)
        return {"from": match.group(1), "to": match.group(2)} if match else {"from": "", "to": ""}

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
    """Látványosságok neveinek kinyerése a felhasználói szövegből."""
    prompt = f"""
    Te egy budapesti turizmusra specializálódott asszisztens vagy.
    A következő szövegből vonj ki minden említett vagy implikált budapesti látványosságot, nevezetességet vagy érdeklődésre számot tartó helyet.
    KIZÁRÓLAG egy JSON tömböt adj vissza a látványosságok neveivel, további szöveg nélkül.
    
    Példa kimenet: ["Parlament", "Budai vár"]
    
    Szöveg: "{text}"
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
    Információk szolgáltatása budapesti látványosságokról webes keresés használatával.
    Args:
        attractions: Látványosságnevek listája, amelyekről információt szeretnénk
    """
    if not attractions or len(attractions) == 0:
        return {"info": "Nincs megadva látványosság.", "source": "web keresés"}
    
    prompt = f"""
Te egy budapesti turisztikai asszisztens vagy.
Kérlek, adj rövid (max 3 mondatos) Budapest-specifikus leírást a következő turisztikai látványosságokról:
{json.dumps(attractions, indent=2)}
KIZÁRÓLAG budapesti kontextusra koncentrálj. Ne adj meg globális vagy irreleváns tartalmat.
Adj vissza egy listát, ahol minden név után következik annak leírása.
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

# === Rendszerprompt az ágenshez ===
prompt = """
Te egy segítőkész magyar asszisztens vagy budapesti tömegközlekedéshez és városnézéshez.
Segítesz turistáknak és helyieknek navigálni Budapesten és érdekes helyeket felfedezni.

Kövesd ezeket a lépéseket a felhasználóknak való válaszadáskor:

1. Útvonaltervezéshez:
   - Vond ki a kiindulási és célállomást a felhasználói bemenetből a parse_input_tool segítségével
   - Hívd meg a directions_tool-t mindkét helyszínnel az útvonal lekéréséhez
   - Formázd az útvonal eredményeit felhasználóbarát összefoglalóvá a következő irányelvek szerint:
     * Kezdd egy fejléccel, amely mutatja a kiindulási hely → célállomás
     * Tartalmazza a teljes időtartamot és távolságot
     * Sorold fel az utazás minden lépését megfelelő ikonokkal:
       - 🚆 tömegközlekedési járművekhez (vonalszámok, járműtípusok és megállónevek mutatása)
       - 🚶 gyaloglási szakaszokhoz (időtartam mutatása)
     * Formázd a lépésszámokat és használj egyértelmű nyilakat (→) a helyszínek között
     * Tömegközlekedési lépéseknél add meg: vonalszám, járműtípus, indulási megálló és érkezési megálló
   - Ha a felhasználó megad egy közlekedési módot (gyaloglás, kerékpározás, autózás), használd azt

2. Látnivalók ajánlásához:
   - Szerezz koordinátákat az útvonal adatokból
   - Hívd meg az attractions_tool-t a releváns koordinátákkal
   - Ha a felhasználó megad egy kategóriát (éttermek, kávézók stb.), használd azt a kategóriát
   - Miután megkaptad a látnivalókat, használd az attraction_info_tool-t pontos leírások beszerzéséhez

3. Konkrét információkért a látnivalókról:
   - Először használd az extract_attractions_tool-t a látnivalónevek azonosításához a kérdésben
   - Ezután használd az attraction_info_tool-t ezekkel a látnivalónevekkel
   - Amikor látnivalókról információkat mutatsz be, EGYÉRTELMŰEN említsd meg, hogy ezt webes keresésből kaptad

FONTOS SZABÁLYOK:
- Amikor a felhasználók budapesti látnivalókról kérdeznek, mindig használd a webes keresési képességet
- Először vond ki a látnivaló neveket az extract_attractions_tool-lal, majd keresd meg őket az attraction_info_tool-lal
- Válaszaidban kifejezetten említsd meg, hogy az információ "webes keresésből" származik
- Az útvonal információk formázásakor a directions_tool-ból, különös figyelmet fordíts a következőkre:
  * Használj következetes, olvasható formátumot megfelelő elrendezéssel
  * Tömegközlekedési útvonalaknál egyértelműen jelöld a vonalszámokat és járműtípusokat (busz, villamos, metró)
  * Használj emojikat a különböző közlekedési módok ábrázolására (🚆, 🚍, 🚇, 🚶)
  * Formázd az időtartamokat és távolságokat olvasható módon
  * Strukturáld a lépésről-lépésre útmutatást egyértelmű számozással
  * Kezeld elegánsan a hibaeseteket (útvonal nem található, érvénytelen helyszínek)
- Mindig fejezd be válaszaidat 1-2 releváns követő kérdéssel a megadott információk alapján:
  * Útvonaltervezésnél: Kérdezz a célállomás közelében lévő látnivalókról vagy éttermekről
  * Látnivaló információknál: Kérdezd meg, hogy szeretnének-e tudni a közeli helyekről vagy hogyan juthatnak oda
  * Általános kérdéseknél: Javasolj kapcsolódó témákat vagy tevékenységeket Budapesten

Mindig magyarul válaszolj, kivéve ha a felhasználó kifejezetten más nyelven kérdez.
Légy segítőkész, barátságos, és adj tömör, de teljes információkat.
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
