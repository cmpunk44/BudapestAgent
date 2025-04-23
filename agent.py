# agent.py

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# === 1. API kulcsok betöltése ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# === 2. LLM példány ===
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# === 3. Tool: helyszínek kinyerése szövegből ===
def parse_trip_input(user_input: str) -> dict:
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
        match = re.search(r'from\s+(.*?)\s+to\s+(.*)', user_input, re.IGNORECASE)
        if match:
            return {"from": match.group(1), "to": match.group(2)}
        # Try Hungarian patterns
        match = re.search(r'(.*?)-(?:ról|ről|ból|ből|tól|től)\s+(?:a |az )?(.*?)(?:-ra|-re|-ba|-be|-hoz|-hez|-höz)?', user_input, re.IGNORECASE)
        return {"from": match.group(1), "to": match.group(2)} if match else {"from": "", "to": ""}

# === 4. Tool: Directions API ===
def get_directions(from_place: str, to_place: str, mode: str = "transit") -> dict:
    """Gets route using Google Directions API with specified transport mode."""
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    # Add Budapest to location if not specified
    if "budapest" not in from_place.lower():
        from_place += ", Budapest, Hungary"
    if "budapest" not in to_place.lower():
        to_place += ", Budapest, Hungary"
        
    params = {
        "origin": from_place,
        "destination": to_place,
        "mode": mode,
        "language": "hu",  # Hungarian language for responses
        "key": MAPS_API_KEY
    }
    
    # Add transit specific parameters if transit mode
    if mode == "transit":
        params["transit_mode"] = "bus|subway|train|tram"
    
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {"error": "Directions API failed"}

# === 5. Tool: Places API with categories ===
def get_local_attractions(lat: float, lng: float, category: str = "tourist_attraction", radius: int = 1000) -> dict:
    """Finds places near coordinates based on specified category."""
    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    # Map user-friendly categories to Google Places API types
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
        "language": "hu",  # Hungarian language
        "key": MAPS_API_KEY
    }
    
    res = requests.get(places_url, params=params)
    if res.status_code == 200:
        data = res.json()
        places = []
        for place in data.get("results", [])[:5]:  # Limit to 5 results
            places.append({
                "name": place.get("name"),
                "rating": place.get("rating", "N/A"),
                "address": place.get("vicinity"),
                "open_now": place.get("opening_hours", {}).get("open_now", "unknown")
            })
        return {"places": places}
    return {"error": "Places API failed", "places": []}

# === 6. Helper function to extract attraction names ===
def extract_attraction_names(text: str) -> list:
    """Extracts potential attraction names from user query."""
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
        # Try to parse JSON array from response
        attractions = json.loads(response.content)
        if isinstance(attractions, list):
            return attractions
    except:
        # If not valid JSON, try to extract from text
        try:
            # Look for anything that appears to be a JSON array
            match = re.search(r'\[(.*?)\]', response.content)
            if match:
                items = match.group(1).split(',')
                return [item.strip(' "\'') for item in items if item.strip()]
        except:
            pass
    
    # Fallback: if specific attractions are mentioned in the original text, use regex
    # to find potential proper nouns (capitalize words)
    potential_attractions = re.findall(r'([A-Z][a-zA-Záéíóöőúüű]+(?:\s+[A-Z][a-zA-Záéíóöőúüű]+)*)', text)
    if potential_attractions:
        return potential_attractions[:3]  # Limit to avoid excessive false positives
    
    return []

# === 7. Reasoning function to analyze user input ===
def analyze_user_intent(text: str) -> Dict[str, Any]:
    """Analyzes user query to determine intent and appropriate tool selection."""
    
    prompt = f"""
    You are a Budapest tourism assistant that analyzes user queries to determine the most appropriate tools to use.
    
    For the following query, analyze what the user wants and suggest the best tools to use.
    Consider these possible intents:
    1. Route planning (user wants to get from place A to place B)
    2. Finding attractions near a location
    3. Information about specific attractions
    4. Finding restaurants/cafes/other venues
    
    For your response, return a JSON object with:
    - "intent": The primary intent of the query
    - "reasoning": Your step-by-step reasoning about why this is the intent
    - "tools": Array of recommended tools to use in order
    - "entities": Any relevant entities extracted (places, attractions, etc.)
    
    Query: "{text}"
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    try:
        # Try to parse JSON from response
        analysis = json.loads(response.content.strip())
        # Ensure required keys exist
        required_keys = ["intent", "reasoning", "tools"]
        if all(key in analysis for key in required_keys):
            return analysis
    except Exception as e:
        print(f"Error parsing analysis: {e}")
        pass
    
    # If parsing fails, return a basic analysis
    return {
        "intent": "unknown",
        "reasoning": "Could not determine intent with confidence.",
        "tools": [],
        "entities": []
    }

# === 8. Reasoning function for next steps ===
def reason_next_steps(context: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes the current state and determines the next step."""
    
    # Extract the conversation history
    messages = context.get("messages", [])
    
    # Get last few messages for context
    last_messages = messages[-10:] if len(messages) > 10 else messages
    
    # Format the messages for the prompt
    formatted_messages = []
    for msg in last_messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_messages.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            formatted_messages.append(f"Tool ({msg.name}): {msg.content[:300]}...")
    
    conversation_context = "\n".join(formatted_messages)
    
    prompt = f"""
    You are a reasoning engine for a Budapest tourism assistant.
    
    Based on the conversation so far and the current state, determine:
    1. What has been accomplished so far
    2. What information is still needed
    3. What should be the next tool to use (if any)
    4. What is the overall plan to complete the user's request
    
    Conversation context:
    {conversation_context}
    
    Return a JSON object with:
    - "progress": Assessment of what's been done so far
    - "reasoning": Your step-by-step reasoning about what to do next
    - "next_tools": Array of recommended tools to use next (if any)
    - "plan": The overall plan to complete the request
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    try:
        # Try to parse JSON from response
        analysis = json.loads(response.content.strip())
        return analysis
    except Exception as e:
        print(f"Error in reason_next_steps: {e}")
        # If parsing fails, return a basic analysis
        return {
            "progress": "In progress",
            "reasoning": "Continuing with the conversation",
            "next_tools": [],
            "plan": "Complete user request"
        }

# === 9. Tool dekorátorok ===
@tool
def reasoning_tool(query: str) -> dict:
    """Analyzes the user's query to determine intent and reasoning.
    Args:
        query: The user's query text
    """
    analysis = analyze_user_intent(query)
    return {
        "analysis_type": "initial_reasoning",
        "intent": analysis.get("intent", "unknown"),
        "reasoning": analysis.get("reasoning", ""),
        "recommended_tools": analysis.get("tools", []),
        "entities": analysis.get("entities", [])
    }

@tool
def next_step_tool(context: str) -> dict:
    """Determines the next steps based on the current conversation state.
    Args:
        context: A description of the current conversation state
    """
    analysis = reason_next_steps({"messages": [HumanMessage(content=context)]})
    return {
        "analysis_type": "next_steps",
        "progress": analysis.get("progress", ""),
        "reasoning": analysis.get("reasoning", ""),
        "next_tools": analysis.get("next_tools", []),
        "plan": analysis.get("plan", "")
    }

@tool
def parse_input_tool(text: str) -> dict:
    """Parses user input and extracts 'from' and 'to' destinations.
    Args:
        text: The user's query text
    """
    return parse_trip_input(text)

@tool
def directions_tool(from_place: str, to_place: str, mode: str = "transit") -> dict:
    """Gets route using Google Directions API.
    Args:
        from_place: Starting location
        to_place: Destination location
        mode: Transportation mode (transit, walking, bicycling, driving)
    """
    return get_directions(from_place, to_place, mode)

@tool
def attractions_tool(lat: float, lng: float, category: str = "tourist_attraction", radius: int = 1000) -> dict:
    """Finds places near coordinates based on category.
    Args:
        lat: Latitude
        lng: Longitude
        category: Place category (attractions, restaurants, cafes, museums, parks, shopping)
        radius: Search radius in meters
    """
    return get_local_attractions(lat, lng, category, radius)

@tool
def extract_attractions_tool(text: str) -> list:
    """Extracts attraction names from the user's query.
    Args:
        text: The user's query text
    """
    return extract_attraction_names(text)

# === Web Search Tool for Attraction Information ===
@tool
def attraction_info_tool(attractions: list) -> dict:
    """
    Provides short Budapest-specific descriptions for a list of attractions using web search.
    Input: list of attraction names (strings).
    Output: dict with "info" containing descriptions of the attractions.
    
    Args:
        attractions: A list of attraction names to get information about
    """
    if not attractions or len(attractions) == 0:
        return {"info": "No attractions specified.", "source": "web search"}
    
    # Log what we're searching for
    print(f"Searching web for information about: {attractions}")
    
    prompt = f"""
You are a tourist assistant specialized in Budapest.
Please provide a short (max 3 sentences) Budapest-specific description for each of the following tourist attractions:
{json.dumps(attractions, indent=2)}
Focus ONLY on Budapest context. No global or irrelevant content.
Return a list where each name is followed by its description.
"""
    try:
        # Use the search-capable model
        gpt4_model = ChatOpenAI(model="gpt-4o-search-preview-2025-03-11", openai_api_key=OPENAI_API_KEY)
        response = gpt4_model.invoke([HumanMessage(content=prompt)])
        
        return {
            "info": response.content,
            "source": "web search",
            "attractions": attractions
        }
    except Exception as e:
        return {
            "info": f"Error retrieving information: {str(e)}",
            "source": "error",
            "attractions": attractions
        }

# === Result formatter tool ===
@tool
def format_route_summary(route_data: Dict[str, Any]) -> str:
    """Formats route data into a user-friendly summary."""
    if not isinstance(route_data, dict):
        try:
            route_data = json.loads(route_data)
        except:
            return "Hibás útvonal adat formátum."
    
    if "error" in route_data:
        return f"Hiba történt: {route_data['error']}"
        
    if "routes" not in route_data or not route_data["routes"]:
        return "Sajnos nem találtam útvonalat."
        
    route = route_data["routes"][0]
    legs = route["legs"][0]
    
    duration = legs["duration"]["text"]
    distance = legs["distance"]["text"]
    
    steps = []
    for step in legs["steps"]:
        if step.get("travel_mode") == "TRANSIT":
            transit = step.get("transit_details", {})
            line = transit.get("line", {}).get("short_name", "")
            vehicle = transit.get("line", {}).get("vehicle", {}).get("name", "jármű")
            departure = transit.get("departure_stop", {}).get("name", "")
            arrival = transit.get("arrival_stop", {}).get("name", "")
            steps.append(f"🚆 {line} {vehicle}: {departure} → {arrival}")
        elif step.get("travel_mode") == "WALKING":
            steps.append(f"🚶 Gyalogolj {step.get('duration', {}).get('text', '')}")
    
    summary = f"""
    🛣️ **Útvonal: {legs['start_address']} → {legs['end_address']}**
    ⏱️ Időtartam: {duration}
    📏 Távolság: {distance}
    
    **Lépések:**
    """
    
    for i, step in enumerate(steps, 1):
        summary += f"{i}. {step}\n"
    
    return summary

# === 10. Custom Message Classes for Reasoning ===
class ReasoningMessage(AIMessage):
    """A special message type to show the agent's reasoning process."""
    
    reasoning_data: Dict[str, Any]
    
    def __init__(self, content: str, reasoning_data: Dict[str, Any]):
        super().__init__(content=content)
        self.reasoning_data = reasoning_data
    
    def __repr__(self):
        return f"ReasoningMessage(content={self.content}, reasoning_data={self.reasoning_data})"

# === 11. AgentState ===
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    reasonings: list[Dict[str, Any]]
    current_reasoning: Optional[Dict[str, Any]]

# === 12. Agent osztály ===
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("reason", self.reason)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        
        # Connect nodes
        graph.add_edge("reason", "llm")
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "reason")  # After taking action, reason again
        
        graph.set_entry_point("reason")
        self.graph = graph.compile()

    def reason(self, state: AgentState) -> AgentState:
        """Reason about the next steps based on the current state."""
        messages = state.get('messages', [])
        reasonings = state.get('reasonings', [])
        
        # If no messages, return empty state
        if not messages:
            return {'messages': messages, 'reasonings': [], 'current_reasoning': None}
        
        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                break
        
        if not last_user_msg:
            return {'messages': messages, 'reasonings': reasonings, 'current_reasoning': None}
        
        # Analyze to determine next step
        if not reasonings:
            # Initial reasoning
            reasoning = analyze_user_intent(last_user_msg.content)
            reasoning_type = "initial_reasoning"
        else:
            # Subsequent reasoning
            state_context = {
                "messages": messages,
                "reasonings": reasonings
            }
            reasoning = reason_next_steps(state_context)
            reasoning_type = "step_reasoning"
        
        # Add timestamp and type
        reasoning["timestamp"] = "now"
        reasoning["type"] = reasoning_type
        
        # Update reasonings list
        updated_reasonings = reasonings + [reasoning]
        
        # Create a reasoning message to add to the conversation
        reasoning_message = f"""
🧠 **Gondolkodási folyamat**:

**Szándék/Haladás**: {reasoning.get('intent', reasoning.get('progress', 'Feldolgozás'))}

**Indoklás**: 
{reasoning.get('reasoning', 'Elemzem a kérést...')}

**Következő lépések**:
{', '.join(reasoning.get('tools', reasoning.get('next_tools', ['Válasz előkészítése'])))}
---
"""
        
        # Add a reasoning message to the conversation
        reasoning_msg = ReasoningMessage(content=reasoning_message, reasoning_data=reasoning)
        updated_messages = messages + [reasoning_msg]
        
        return {
            'messages': updated_messages, 
            'reasonings': updated_reasonings,
            'current_reasoning': reasoning
        }

    def exists_action(self, state: AgentState):
        if not state['messages']:
            return False
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(getattr(result, 'tool_calls', [])) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        reasonings = state.get('reasonings', [])
        current_reasoning = state.get('current_reasoning')
        
        # Filter out ReasoningMessages before sending to the model
        filtered_messages = [msg for msg in messages if not isinstance(msg, ReasoningMessage)]
        
        # Add system message if not present
        if self.system and not any(isinstance(msg, SystemMessage) for msg in filtered_messages):
            system_content = self.system
            
            # If we have current reasoning, include it in the system message
            if current_reasoning:
                reasoning_str = json.dumps(current_reasoning, indent=2)
                system_content += f"\n\nCurrent reasoning analysis:\n{reasoning_str}\n\n"
                system_content += "Use this reasoning to guide your tool selection and response strategy."
                
            filtered_messages = [SystemMessage(content=system_content)] + filtered_messages
        
        message = self.model.invoke(filtered_messages)
        return {'messages': [message], 'reasonings': reasonings, 'current_reasoning': current_reasoning}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        reasonings = state.get('reasonings', [])
        current_reasoning = state.get('current_reasoning')
        
        results = []
        for t in tool_calls:
            if t['name'] not in self.tools:
                result = f"Invalid tool name: {t['name']}. Retry."
            else:
                try:
                    result = self.tools[t['name']].invoke(t['args'])
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
        return {'messages': results, 'reasonings': reasonings, 'current_reasoning': current_reasoning}

# === 13. Agent példány kibővített prompttal ===
prompt = """
You are a helpful Hungarian assistant for Budapest public transport and sightseeing.
You help tourists and locals navigate Budapest and discover interesting places.

Follow the reasoning process and use the recommended tools. Your task is to:

1. For route planning:
   - Extract origin and destination from user input using parse_input_tool
   - Call directions_tool with both locations to get a route
   - Format the route results in a user-friendly way with format_route_summary
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
- Follow the reasoning process to determine which tools to use
- When users ask about attractions in Budapest, always use the web search capability
- First extract attraction names with extract_attractions_tool, then look them up with attraction_info_tool
- Explicitly state that information comes from "web search" in your responses
   
Always respond in Hungarian unless the user specifically asks in another language.
Be helpful, friendly, and provide concise but complete information.
"""

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
tools = [
    reasoning_tool,
    next_step_tool,
    parse_input_tool, 
    directions_tool, 
    attractions_tool,
    extract_attractions_tool,
    attraction_info_tool,
    format_route_summary
]
budapest_agent = Agent(model, tools, system=prompt)
