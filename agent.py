from dotenv import load_dotenv
load_dotenv()

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
from gtfs_loader import gtfs

# === 1. API kulcsok betöltése ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# === 2. LLM példány ===
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# === 3. Tool: helyszínek kinyerése szövegből ===
def parse_trip_input(user_input: str) -> dict:
    prompt = f"""
    You are a multilingual assistant. Extract two locations from this sentence.
    Respond ONLY with a JSON like:
    {{"from": "X", "to": "Y"}}
    Input: "{user_input}"
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    try:
        return json.loads(response.content)
    except:
        match = re.search(r'from\s+(.*?)\s+to\s+(.*)', user_input, re.IGNORECASE)
        return {"from": match.group(1), "to": match.group(2)} if match else {"from": "", "to": ""}

# === 4. Tool: Directions API ===
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

# === 5. Tool: Places API ===
def get_local_attractions(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict:
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    attractions = []
    for lat, lng in [(start_lat, start_lng), (end_lat, end_lng)]:
        params = {
            "location": f"{lat},{lng}",
            "radius": 1000,
            "type": "tourist_attraction",
            "key": MAPS_API_KEY
        }
        res = requests.get(places_url, params=params)
        if res.status_code == 200:
            data = res.json()
            attractions += [r.get("name") for r in data.get("results", [])]
    return {"attractions": attractions}

# === 6. Tool: GPT-4o leírás attrakciókhoz (javított input) ===
@tool
def attraction_info_tool(attractions: list) -> dict:
    """
    Provides short Budapest-specific descriptions for a list of attractions.
    Input: list of attraction names (strings).
    Output: dict with name → description pairs.
    """
    if not attractions:
        return {"info": {}}

    prompt = f"""
You are a tourist assistant specialized in Budapest.

Please provide a short (max 3 sentences) Budapest-specific description for each of the following tourist attractions:

{json.dumps(attractions, indent=2)}

Focus ONLY on Budapest context. No global or irrelevant content.
Return a list where each name is followed by its description.
"""

    gpt4_model = ChatOpenAI(model="gpt-4o-search-preview-2025-03-11")
    response = gpt4_model.invoke([HumanMessage(content=prompt)])
    return {"info": response.content}

from google.transit import gtfs_realtime_pb2
import pandas as pd
import time

@tool
def get_schedule_tool(route_name: str, stop_name: str) -> dict:
    """
    Returns the next 5 real-time departure times for a given BKK route and stop.
    Requires GTFS static files: routes.txt, stops.txt
    Requires GTFS-RT API key in .env as BKK_API_KEY
    """
    try:
        # Load GTFS static data
        routes_df = pd.read_csv("routes.txt")
        stops_df = pd.read_csv("stops.txt")

        # Match route_id from short name
        route_row = routes_df[routes_df["route_short_name"] == route_name]
        if route_row.empty:
            return {"error": f"No GTFS route found with short name '{route_name}'."}
        route_id = route_row["route_id"].values[0]

        # Match stop_id from stop name
        stop_row = stops_df[stops_df["stop_name"] == stop_name]
        if stop_row.empty:
            return {"error": f"No GTFS stop found with name '{stop_name}'."}
        stop_id = stop_row["stop_id"].values[0]

        # Load GTFS-realtime TripUpdates
        api_key = os.getenv("BKK_API_KEY")
        if not api_key:
            return {"error": "Missing BKK_API_KEY in environment."}

        url = f"https://go.bkk.hu/api/query/v1/ws/gtfs-rt/full/TripUpdates.pb?key={api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": "GTFS-realtime API request failed."}

        # Parse the binary protobuf feed
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)

        # Collect departure times
        departures = []
        for entity in feed.entity:
            if entity.HasField("trip_update"):
                trip = entity.trip_update
                if trip.trip.route_id == route_id:
                    for stu in trip.stop_time_update:
                        if stu.stop_id == stop_id and stu.HasField("departure"):
                            ts = stu.departure.time
                            departures.append(time.strftime('%H:%M', time.localtime(ts)))

        if not departures:
            return {"message": f"No departures currently available for {route_name} at {stop_name}."}

        return {
            "route": route_name,
            "stop": stop_name,
            "next_departures": departures[:5]
        }

    except Exception as e:
        return {"error": str(e)}

# === 7. Tool dekorátorok ===
@tool
def parse_input_tool(text: str) -> dict:
    """Parses user input and extracts 'from' and 'to' destinations."""
    return parse_trip_input(text)

@tool
def directions_tool(from_place: str, to_place: str) -> dict:
    """Gets public transport route using Google Directions API."""
    return get_directions(from_place, to_place)

@tool
def attractions_tool(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict:
    """Finds tourist attractions near the route using Google Places API."""
    return get_local_attractions(start_lat, start_lng, end_lat, end_lng)

# === 8. AgentState ===
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# === 9. Agent osztály ===
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.history = []

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

    # === Chat history kezelés ===
    def add_user_message(self, message: str):
        self.history.append(HumanMessage(content=message))

    def reset_history(self):
        self.history = []

    def get_history(self) -> list:
        return self.history

    def run(self):
        return self.graph.invoke({"messages": self.history})

# === 10. Agent példány létrehozása ===
prompt = """
You are a helpful assistant for Budapest public transport and sightseeing.

You can:
- Parse origin and destination from user input
- Call directions_tool with both locations to get route
- Call attractions_tool with coordinates extracted from route_data (start and end lat/lng)
- If the user asks for more information about specific attractions, use attraction_info_tool with the attraction list.
- You can call get_schedule_tool when the user asks about departure times or schedule information (e.g., "When does tram 4 leave from Móricz on Monday?").

Always focus on Budapest. Never include information about locations outside of Budapest.
"""

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
tools = [parse_input_tool, directions_tool, attractions_tool, attraction_info_tool]
budapest_agent = Agent(model, tools, system=prompt)
