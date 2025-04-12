# === ReAct-style agent.py ===

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

# === API kulcsok ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# === LLM példány ===
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# === Tool: helyszínek kinyerése szövegből ===
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

# === Tool: Directions API ===
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

# === Tool: Places API ===
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

# === Tool dekorátorok ===
@tool
def parse_input_tool(text: str) -> dict:
    return parse_trip_input(text)

@tool
def directions_tool(from_place: str, to_place: str) -> dict:
    return get_directions(from_place, to_place)

@tool
def attractions_tool(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict:
    return get_local_attractions(start_lat, start_lng, end_lat, end_lng)

# === AgentState ===
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# === ReAct-stílusú Agent ===
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

        messages = state["messages"] + results
        next_llm = self.model.invoke(messages)
        return {"messages": results + [next_llm]}

# === Prompt ===
prompt = """
You are a ReAct-style assistant. Help the user plan a public transport route in Budapest and recommend nearby tourist attractions.

For each step:
- Thought: explain what you're thinking
- Action: call one tool
- Observation: process result
- Repeat until ready

Tools:
- parse_input_tool(text: str) → {'from': ..., 'to': ...}
- directions_tool(from_place, to_place)
- attractions_tool(start_lat, start_lng, end_lat, end_lng)

When done:
Thought: I now have enough information
Final Answer: [...]
"""

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
tools = [parse_input_tool, directions_tool, attractions_tool]
budapest_agent = Agent(model, tools, system=prompt)
