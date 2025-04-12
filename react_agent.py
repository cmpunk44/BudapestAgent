# react_agent.py

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent

# === API Keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# === Tools ===
@tool
def directions_tool(from_place: str, to_place: str) -> dict:
    """Gets public transport directions between two places using Google Maps API."""
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": from_place,
        "destination": to_place,
        "mode": "transit",
        "transit_mode": "bus|subway|train|tram",
        "key": MAPS_API_KEY
    }
    res = requests.get(url, params=params)
    return res.json() if res.status_code == 200 else {"error": "Directions API failed"}

@tool
def attraction_tool(lat: float, lng: float) -> dict:
    """Finds tourist attractions near a single location using Google Places API."""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": 1000,
        "type": "tourist_attraction",
        "key": MAPS_API_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        data = res.json()
        return {"attractions": [r.get("name") for r in data.get("results", [])]}
    else:
        return {"error": "Places API failed"}

# === Model & Agent ===
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)
tools = [directions_tool, attraction_tool]

react_agent = create_react_agent(llm=model, tools=tools)
react_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
