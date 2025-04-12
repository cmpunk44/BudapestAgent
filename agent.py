# --- Budapest Agent with simplified attraction logic (start, end, transfers only) ---

import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# --- API kulcsok betöltése ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# --- LLM példány ---
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# --- Tool: helyszínek kinyerése szövegből ---
def parse_trip_input(user_input: str) -> dict:
    prompt = f'''
    You are a multilingual assistant.
    Extract the origin and destination from the following sentence.

    Return a JSON object ONLY like this:
    {{"from": "starting location", "to": "destination"}}

    Examples:
    - "How do I get from Blaha Lujza tér to Gellért Hill?"
      → {{"from": "Blaha Lujza tér", "to": "Gellért Hill"}}
    - "Utazzunk el a Széll Kálmán térről a Fővám térre!"
      → {{"from": "Széll Kálmán tér", "to": "Fővám tér"}}

    Sentence: "{user_input}"
    '''
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    try:
        return json.loads(response.content)
    except:
        match = re.search(r'from\s+(.*?)\s+to\s+(.*)', user_input, re.IGNORECASE)
        return {"from": match.group(1), "to": match.group(2)} if match else {"from": "", "to": ""}

# --- Tool: Directions API ---
def get_directions(from_place: str, to_place: str) -> dict:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": from_place,
        "destination": to_place,
        "mode": "transit",
        "transit_mode": "bus|subway|train|tram",
        "key": MAPS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {"error": "Directions API failed"}

# --- Egyszerűsített attrakciókeresés (start, end, transfer) ---
def get_main_point_attractions(route_data: dict) -> dict:
    try:
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        coords = []

        legs = route_data.get("routes", [])[0].get("legs", [])
        if not legs:
            return {"attractions": []}

        # Start and end point
        coords.append(legs[0].get("start_location"))
        coords.append(legs[-1].get("end_location"))

        # Transfer points (if any)
        for step in legs[0].get("steps", []):
            if step.get("travel_mode") == "TRANSIT":
                transit = step.get("transit_details", {})
                transfer_stop = transit.get("arrival_stop", {}).get("location")
                if transfer_stop:
                    coords.append(transfer_stop)

        # Search attractions near these points
        attractions = set()
        for loc in coords:
            if not loc:
                continue
            lat, lng = loc.get("lat"), loc.get("lng")
            if lat is None or lng is None:
                continue
            params = {
                "location": f"{lat},{lng}",
                "radius": 1000,
                "type": "tourist_attraction",
                "key": MAPS_API_KEY
            }
            res = requests.get(places_url, params=params)
            if res.status_code == 200:
                data = res.json()
                names = [r.get("name") for r in data.get("results", []) if r.get("name")]
                attractions.update(names)

        return {"attractions": list(attractions)}
    except Exception as e:
        return {"error": str(e)}

# --- Tool dekorátorok ---
@tool
def parse_input_tool(text: str) -> dict:
    """Parses user input and extracts 'from' and 'to' destinations."""
    return parse_trip_input(text)

@tool
def directions_tool(from_place: str, to_place: str) -> dict:
    """Gets public transport route using Google Directions API."""
    return get_directions(from_place, to_place)

@tool
def get_main_point_attractions_tool(*, route_data: dict) -> dict:
    """Finds tourist attractions at start, end, and transfer points."""
    return get_main_point_attractions(route_data)

# --- AgentState definíció ---
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# --- Agent osztály ---
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] not in self.tools:
                result = "Invalid tool name. Retry."
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

# --- Agent példány ---
prompt = """
You are a helpful assistant specialized in Budapest public transport and sightseeing. Your job is to help the user travel across the city and discover tourist attractions.

Workflow:
1. Use parse_input_tool to extract origin and destination.
2. Use directions_tool to get a route.
3. Use get_main_point_attractions_tool and pass the full route_data.
   Only find attractions near the starting location, ending location, and transfer stops.

Call all tools explicitly using the correct tool name and argument keys.
"""

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
tools = [parse_input_tool, directions_tool, get_main_point_attractions_tool]
budapest_agent = Agent(model, tools, system=prompt)
