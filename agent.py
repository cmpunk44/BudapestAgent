# agent.py
# Simple LangGraph-based agent for Budapest tourism and transit information
# Author: Szalay Miklós Márton
# Thesis project for Pannon University

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated, List, Dict, Any

# Import necessary LangChain components
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# Initialize the LLM with OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.3)

# === Tool functions ===

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
        # Fallback parsing with simple regex
        match = re.search(r'from\s+(.*?)\s+to\s+(.*)', user_input, re.IGNORECASE)
        if match:
            return {"from": match.group(1), "to": match.group(2)}
        # Try Hungarian patterns
        match = re.search(r'(.*?)-(?:ról|ről|ból|ből|tól|től)\s+(?:a |az )?(.*?)(?:-ra|-re|-ba|-be|-hoz|-hez|-höz)?', user_input, re.IGNORECASE)
        return {"from": match.group(1), "to": match.group(2)} if match else {"from": "", "to": ""}

def get_directions(from_place: str, to_place: str, mode: str = "transit") -> dict:
    """Get route directions using Google Directions API."""
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

def get_local_attractions(lat: float, lng: float, category: str = "tourist_attraction", radius: int = 1000) -> dict:
    """Find places near coordinates based on category using Google Places API."""
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
        "language": "hu",
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
        # Try to parse JSON array from response
        attractions = json.loads(response.content)
        if isinstance(attractions, list):
            return attractions
    except:
        # Fallback: Find potential proper nouns (capitalized words)
        potential_attractions = re.findall(r'([A-Z][a-zA-Záéíóöőúüű]+(?:\s+[A-Z][a-zA-Záéíóöőúüű]+)*)', text)
        if potential_attractions:
            return potential_attractions[:3]  # Limit to avoid excessive false positives
    
    return []

def generate_followup_suggestions(context: str, query_type: str, destination: str = None) -> list:
    """Generate relevant follow-up questions based on the conversation context."""
    prompt = f"""
    You are a Budapest tourism assistant helping users explore the city.
    Based on the current conversation and the user's query type, generate 2-3 natural follow-up questions 
    that would help the user discover more about Budapest.
    
    Context: {context}
    Query type: {query_type}
    Destination: {destination if destination else "Not specified"}
    
    For each question type, focus on:
    - Route planning: Ask about attractions near the destination or transportation preferences
    - Attractions: Ask about specific interests (history, architecture, views)
    - Food/restaurants: Ask about cuisine preferences or dietary restrictions
    - General: Ask about duration of stay or specific interests
    
    Return ONLY a JSON array of follow-up questions in Hungarian, with no additional text.
    Example: ["Szeretnél megtudni valamit a környékbeli látnivalókról?", "Érdekelnének étkezési lehetőségek az útvonalon?"]
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    try:
        # Try to parse JSON array from response
        suggestions = json.loads(response.content)
        if isinstance(suggestions, list):
            return suggestions[:2]  # Limit to 2 suggestions
    except:
        # Fallback with regex
        suggestions = re.findall(r'"([^"]+)"', response.content)
        if suggestions:
            return suggestions[:2]
    
    # Default suggestions if parsing fails
    default_suggestions = [
        "Szeretnél többet megtudni a környékbeli látnivalókról?",
        "Segíthetek még valamiben Budapest felfedezésében?"
    ]
    
    return default_suggestions

# === Register tools with LangChain's @tool decorator ===

@tool
def parse_input_tool(text: str) -> dict:
    """Parses user input and extracts 'from' and 'to' destinations."""
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

@tool
def followup_suggestions_tool(context: str, query_type: str, destination: str = None) -> list:
    """Generates follow-up questions based on the conversation context.
    Args:
        context: The conversation context
        query_type: The type of query (route, attraction, food, etc.)
        destination: Optional destination mentioned in the conversation
    """
    return generate_followup_suggestions(context, query_type, destination)

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

# === Define the agent state ===
class AgentState(TypedDict):
    """Represents the state of the agent throughout the conversation."""
    messages: Annotated[list[AnyMessage], operator.add]  # The messages accumulate
    query_type: str  # Type of the current query (route, attraction, etc.)
    destination: str  # Destination mentioned in the query, if any

# === Agent class to manage the conversation flow ===
class Agent:
    """A LangGraph-based agent that can use tools to help with Budapest tourism queries."""
    
    def __init__(self, model, tools, system=""):
        """Initialize the agent with a language model, tools, and system prompt."""
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        # Create a graph with nodes for different stages
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("analyze", self.analyze_query)  # New node to analyze query
        graph.add_node("llm", self.call_openai)  # Node for generating responses or tool calls
        graph.add_node("action", self.take_action)  # Node for executing tools
        graph.add_node("followup", self.add_followups)  # New node to add follow-up suggestions
        
        # Add edges to define the flow
        graph.add_edge("analyze", "llm")  # First analyze, then call LLM
        
        graph.add_conditional_edges(
            "llm",  # From the LLM node
            self.exists_action,  # Check if there's a tool to call
            {True: "action", False: "followup"}  # If yes, go to action; if no, add followups
        )
        
        graph.add_edge("action", "llm")  # After action, go back to LLM
        graph.add_edge("followup", END)  # After adding followups, end
        
        # Set the entry point
        graph.set_entry_point("analyze")
        
        # Compile the graph
        self.graph = graph.compile()

    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to determine the type and extract key information."""
        messages = state.get('messages', [])
        
        # Default values
        query_type = "general"
        destination = ""
        
        # Find the last user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if user_message:
            # Simple analysis to categorize query type
            lower_msg = user_message.lower()
            
            # Check for route planning queries
            if any(word in lower_msg for word in ["juthatok", "eljutni", "útvonal", "hogyan jutok", "hogyan menjek", "útiterv"]):
                query_type = "route"
                
                # Try to extract destination
                try:
                    places = parse_trip_input(user_message)
                    if places and "to" in places:
                        destination = places["to"]
                except:
                    pass
            
            # Check for attraction queries
            elif any(word in lower_msg for word in ["látnivaló", "látnivalók", "látogassak", "múzeum", "érdekes hely"]):
                query_type = "attraction"
                
                # Try to extract attraction names
                try:
                    attractions = extract_attraction_names(user_message)
                    if attractions:
                        destination = attractions[0]
                except:
                    pass
            
            # Check for food/restaurant queries
            elif any(word in lower_msg for word in ["étterem", "kávézó", "enni", "étel", "kávé", "ebéd", "vacsora"]):
                query_type = "food"
        
        # Return updated state with query analysis
        return {
            'messages': messages,
            'query_type': query_type,
            'destination': destination
        }

    def exists_action(self, state: AgentState) -> bool:
        """Check if the last message contains any tool calls."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(getattr(result, 'tool_calls', [])) > 0

    def call_openai(self, state: AgentState) -> AgentState:
        """Call the language model to generate a response or tool calls."""
        messages = state['messages']
        query_type = state.get('query_type', 'general')
        destination = state.get('destination', '')
        
        # Add system message if not present
        if self.system and not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=self.system)] + messages
            
        # Call the model and get a response
        message = self.model.invoke(messages)
        
        # Return the updated state with the new message
        return {
            'messages': [message],
            'query_type': query_type,
            'destination': destination
        }

    def take_action(self, state: AgentState) -> AgentState:
        """Execute any tool calls from the language model."""
        tool_calls = state['messages'][-1].tool_calls
        query_type = state.get('query_type', 'general')
        destination = state.get('destination', '')
        results = []
        
        # Process each tool call
        for t in tool_calls:
            if t['name'] not in self.tools:
                result = f"Invalid tool name: {t['name']}. Retry."
            else:
                try:
                    # Call the tool with the arguments
                    result = self.tools[t['name']].invoke(t['args'])
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
                    
            # Create a tool message with the result
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            
        # Return the updated state with the tool results
        return {
            'messages': results,
            'query_type': query_type,
            'destination': destination
        }
    
    def add_followups(self, state: AgentState) -> AgentState:
        """Add follow-up questions to the final response."""
        messages = state['messages']
        query_type = state.get('query_type', 'general')
        destination = state.get('destination', '')
        
        # Get the last message which should be the final response
        if not messages or not isinstance(messages[-1], SystemMessage):
            # Try to generate some follow-up suggestions
            try:
                # Create context from the last few messages
                context = ""
                for msg in messages[-3:]:  # Last 3 messages
                    if hasattr(msg, 'content'):
                        context += msg.content + "\n"
                
                # Generate follow-up suggestions
                suggestions = generate_followup_suggestions(context, query_type, destination)
                
                # Get the last message
                last_message = messages[-1]
                
                if isinstance(last_message, SystemMessage) or not hasattr(last_message, 'content'):
                    # Can't modify this message type, just return
                    return {'messages': messages, 'query_type': query_type, 'destination': destination}
                
                # Add suggestions to the response content
                content = last_message.content
                
                # Add a section for follow-up questions
                content += "\n\n---\n**További kérdések:**\n"
                for i, suggestion in enumerate(suggestions, 1):
                    content += f"{i}. {suggestion}\n"
                
                # Create a new message with the updated content
                if hasattr(last_message, '__class__'):
                    # Try to create same message type
                    new_message = last_message.__class__(content=content)
                else:
                    # Fallback to system message
                    new_message = SystemMessage(content=content)
                
                # Replace the last message with our enhanced version
                messages[-1] = new_message
                
            except Exception as e:
                print(f"Error adding followups: {e}")
                # If something goes wrong, just return the original messages
                pass
        
        return {'messages': messages, 'query_type': query_type, 'destination': destination}

# === System prompt for the agent ===
prompt = """
You are a helpful Hungarian assistant for Budapest public transport and sightseeing.
You help tourists and locals navigate Budapest and discover interesting places.

Follow these steps when responding to users:

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
- When users ask about attractions in Budapest, always use the web search capability
- First extract attraction names with extract_attractions_tool, then look them up with attraction_info_tool
- Explicitly state that information comes from "web search" in your responses
- After providing an answer, encourage the user to ask follow-up questions about nearby attractions, food options, or other points of interest
   
Always respond in Hungarian unless the user specifically asks in another language.
Be helpful, friendly, and provide concise but complete information.
"""

# Create the model instance
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Define the tools available to the agent
tools = [
    parse_input_tool, 
    directions_tool, 
    attractions_tool,
    extract_attractions_tool,
    attraction_info_tool,
    followup_suggestions_tool,
    format_route_summary
]

# Create the agent instance
budapest_agent = Agent(model, tools, system=prompt)
