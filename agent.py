# agent.py

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple, Union, Literal

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

# === 7. Tool dekorátorok ===
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

# === 8. Custom ThoughtState and Message Class ===
class ThoughtState(TypedDict):
    """Represents the agent's current state of thinking."""
    thought: str
    action_plan: Optional[List[str]]
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]

class ThoughtMessage(AIMessage):
    """A message representing the agent's internal thought process."""
    def __init__(self, content: str, thinking: Dict[str, Any] = None):
        super().__init__(content=content)
        self.thinking = thinking if thinking else {}

# === 9. AgentState ===
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    thoughts: List[ThoughtState]
    current_thought: Optional[ThoughtState]

# === 10. Agent osztály ===
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        
        # Define the state machine
        graph = StateGraph(AgentState)
        
        # Add nodes to the graph
        graph.add_node("think", self.think)
        graph.add_node("act", self.act)
        
        # Add edges to connect the nodes
        graph.add_edge("think", "act")
        graph.add_conditional_edges("act", self.should_continue, {True: "think", False: END})
        
        # Set the entry point to start with thinking
        graph.set_entry_point("think")
        
        self.graph = graph.compile()
    
    def think(self, state: AgentState) -> AgentState:
        """Think about what to do next based on the context."""
        messages = state.get('messages', [])
        thoughts = state.get('thoughts', [])
        
        # Extract the last user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            # No user message to respond to
            return {'messages': messages, 'thoughts': thoughts, 'current_thought': None}
        
        # Get the most recent non-thought message exchange to consider
        relevant_context = []
        for msg in messages:
            if not isinstance(msg, ThoughtMessage):
                relevant_context.append(msg)
        
        # Create a thinking prompt to make the agent analyze the situation
        thinking_prompt = f"""
        As a Budapest tourism assistant, think step by step about how to address this request.
        
        User request: "{user_message}"
        
        Think about:
        1. What is the user asking for? (route planning, attraction information, recommendations, etc.)
        2. What information do I need to gather?
        3. What tools should I use in what order?
        4. What's my step-by-step plan?
        
        Format your thinking as:
        
        Thought: [Your detailed analysis of the request]
        
        Action Plan: [The sequence of actions/tools you'll use]
        
        Action: [The specific next tool to use]
        
        Action Input: [The exact parameters to pass to the tool]
        """
        
        # Add system prompt for thinking
        thinking_messages = [
            SystemMessage(content="You are a helpful assistant planning how to handle a user request about Budapest tourism."),
            HumanMessage(content=thinking_prompt)
        ]
        
        # Get the agent's thinking response
        thinking_response = llm.invoke(thinking_messages)
        thinking_content = thinking_response.content
        
        # Parse the thinking response to extract structured thinking
        thought_state = self._parse_thinking(thinking_content)
        
        # Create a human-readable format of the thinking for display
        readable_thinking = f"""
        # 🧠 Gondolkodási folyamat:
        
        **Elemzés:**
        {thought_state.get('thought', 'No analysis provided')}
        
        **Cselekvési terv:**
        {", ".join(thought_state.get('action_plan', ['No plan provided']))}
        
        **Következő lépés:**
        {thought_state.get('action', 'No action specified')} {json.dumps(thought_state.get('action_input', {}), ensure_ascii=False)}
        """
        
        # Create a thought message to add to the conversation
        thought_message = ThoughtMessage(content=readable_thinking, thinking=thought_state)
        
        # Update thoughts list
        updated_thoughts = thoughts + [thought_state]
        
        # Add thought message to conversation
        updated_messages = messages + [thought_message]
        
        return {
            'messages': updated_messages,
            'thoughts': updated_thoughts,
            'current_thought': thought_state
        }
    
    def _parse_thinking(self, thinking_content: str) -> ThoughtState:
        """Parse the thinking content into structured parts."""
        thought_state = {
            "thought": "",
            "action_plan": [],
            "action": None,
            "action_input": {}
        }
        
        # Extract thought
        thought_match = re.search(r'Thought:(.*?)(?:Action Plan:|Action:|$)', thinking_content, re.DOTALL)
        if thought_match:
            thought_state["thought"] = thought_match.group(1).strip()
        
        # Extract action plan
        plan_match = re.search(r'Action Plan:(.*?)(?:Action:|$)', thinking_content, re.DOTALL)
        if plan_match:
            plan_text = plan_match.group(1).strip()
            # Split by numbered items or commas or new lines
            items = re.split(r'\n+|\d+\.\s+|,\s*', plan_text)
            thought_state["action_plan"] = [item.strip() for item in items if item.strip()]
        
        # Extract action
        action_match = re.search(r'Action:(.*?)(?:Action Input:|$)', thinking_content, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            # Try to match against available tool names
            for tool_name in self.tools.keys():
                if tool_name.lower() in action.lower():
                    thought_state["action"] = tool_name
                    break
            
            # If no match found, use the text as is
            if not thought_state["action"]:
                thought_state["action"] = action
        
        # Extract action input
        input_match = re.search(r'Action Input:(.*?)$', thinking_content, re.DOTALL)
        if input_match:
            input_text = input_match.group(1).strip()
            
            # Try to parse as JSON
            try:
                thought_state["action_input"] = json.loads(input_text)
            except:
                # Try to extract key-value pairs
                pairs = re.findall(r'(\w+):\s*([^,\n]+)', input_text)
                if pairs:
                    thought_state["action_input"] = {k.strip(): v.strip() for k, v in pairs}
                else:
                    # Use the entire text as a single parameter
                    action = thought_state["action"]
                    if action in self.tools:
                        # Check if the tool expects a single string parameter
                        thought_state["action_input"] = {"text": input_text}
        
        return thought_state
    
    def act(self, state: AgentState) -> AgentState:
        """Take action based on the current thought."""
        messages = state.get('messages', [])
        thoughts = state.get('thoughts', [])
        current_thought = state.get('current_thought')
        
        if not current_thought or not current_thought.get('action'):
            # Create a response message if we can't determine an action
            response = "I'm not sure how to proceed with your request. Could you provide more details?"
            response_message = AIMessage(content=response)
            return {'messages': messages + [response_message], 'thoughts': thoughts, 'current_thought': None}
        
        action = current_thought.get('action')
        action_input = current_thought.get('action_input', {})
        
        # Check if the action is a valid tool
        if action in self.tools:
            try:
                # Call the tool with the provided input
                result = self.tools[action].invoke(action_input)
                # Create a tool message with the result
                tool_message = ToolMessage(tool_call_id="1", name=action, content=str(result))
                return {'messages': messages + [tool_message], 'thoughts': thoughts, 'current_thought': current_thought}
            except Exception as e:
                # If tool execution fails, create an error message
                error_message = f"Error executing tool {action}: {str(e)}"
                tool_message = ToolMessage(tool_call_id="1", name=action, content=error_message)
                return {'messages': messages + [tool_message], 'thoughts': thoughts, 'current_thought': current_thought}
        else:
            # If the action is not a tool, treat it as a response
            if action.lower() == "respond" or action.lower() == "answer":
                # Use the action_input as the response content if available
                content = action_input.get('text', "I'll provide an answer based on what I know.")
                response_message = AIMessage(content=content)
                return {'messages': messages + [response_message], 'thoughts': thoughts, 'current_thought': None}
            else:
                # Create final response based on all gathered information
                system_prompt = self.system if self.system else "You are a helpful Hungarian assistant for Budapest tourism."
                prompt_messages = [SystemMessage(content=system_prompt)]
                
                # Add relevant conversation history (filter out thinking messages)
                for msg in messages:
                    if not isinstance(msg, ThoughtMessage):
                        prompt_messages.append(msg)
                
                # Add a prompt to generate the final response
                prompt_messages.append(HumanMessage(content="Based on all the information gathered, provide a comprehensive response to the user's query."))
                
                # Generate response
                response = self.model.invoke(prompt_messages)
                return {'messages': messages + [response], 'thoughts': thoughts, 'current_thought': None}
    
    def should_continue(self, state: AgentState) -> bool:
        """Determine if we should continue thinking or conclude the interaction."""
        messages = state.get('messages', [])
        
        # Get the last message
        if not messages:
            return False
            
        last_message = messages[-1]
        
        # If the last message is a tool message, we should continue thinking
        if isinstance(last_message, ToolMessage):
            return True
            
        # If it's an AI message (a final response), we're done
        if isinstance(last_message, AIMessage) and not isinstance(last_message, ThoughtMessage):
            return False
            
        # If it's a thought message, we need to act on it
        if isinstance(last_message, ThoughtMessage):
            return True
            
        # By default, continue
        return True

# === 11. Agent példány ===
prompt = """
You are a helpful Hungarian assistant for Budapest public transport and sightseeing.
You help tourists and locals navigate Budapest and discover interesting places.

Your capabilities include:
1. Route planning with Budapest public transport
   - Use parse_input_tool to extract locations
   - Use directions_tool to find routes
   - Use format_route_summary to present routes clearly

2. Finding attractions near locations
   - Use attractions_tool to find places of interest
   - Recommend places based on user preferences

3. Providing information about Budapest landmarks
   - Use extract_attractions_tool to identify attractions
   - Use attraction_info_tool to get accurate information from the web
   - Always prefer web search over your general knowledge

Always respond in Hungarian unless the user specifically asks in another language.
Be helpful, friendly, and provide concise but complete information.
"""

model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
tools = [
    parse_input_tool, 
    directions_tool, 
    attractions_tool,
    extract_attractions_tool,
    attraction_info_tool,
    format_route_summary
]
budapest_agent = Agent(model, tools, system=prompt)
