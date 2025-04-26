# agent.py
# LangGraph-based agent for Budapest tourism and transit information
# Simplified finalization approach
# Author: Szalay Miklós Márton
# Modified by: Claude 3.7 Sonnet

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import json
import re
import requests
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Sequence, Union, Optional

# Import necessary LangChain components
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage, AIMessage
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

# === Define the agent state ===
class AgentState(TypedDict):
    """Represents the state of the agent throughout the conversation."""
    messages: Annotated[list[AnyMessage], operator.add]  # The messages accumulate
    reasoning_output: Optional[str]  # The reasoning behind the current decision
    next_step: Optional[str]  # What action to take next
    tool_history: Annotated[list[Dict], operator.add]  # Track tool calls and results for final response
    has_used_tools: Optional[bool]  # Track if any tools have been used
    user_query: Optional[str]  # Store the original user query

# === Agent class to manage the conversation flow ===
class Agent:
    """A LangGraph-based agent that can use tools to help with Budapest tourism queries."""
    
    def __init__(self, model, tools, system=""):
        """Initialize the agent with a language model, tools, and system prompt."""
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        
        # Create reason model without tools binding
        self.reason_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.1)
        
        # Create finalize model without tools binding
        self.finalize_model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.7)

        # Create a graph with four nodes: reasoning, LLM, action, and finalize
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("reason", self.reason_about_request)  # Node for reasoning about the request
        graph.add_node("llm", self.call_openai)  # Node for generating tool calls
        graph.add_node("action", self.take_action)  # Node for executing tools
        graph.add_node("finalize", self.finalize_response)  # Node for generating final response
        
        # Add conditional edges
        graph.add_conditional_edges(
            "reason",
            self.determine_next_step,
            {
                "use_tools": "llm",  # If tools are needed, go to LLM
                "respond_directly": END  # If no tools needed, go to end
            }
        )
        
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {
                True: "action",  # If there are tool calls, go to action
                False: "finalize"  # If no tool calls, go to finalize
            }
        )
        
        graph.add_conditional_edges(
            "action",
            self.should_continue,
            {
                True: "llm",  # If need more tools, go back to LLM
                False: "finalize"  # If we're done with tools, go to finalize
            }
        )
        
        # After finalize, always end
        graph.add_edge("finalize", END)
        
        # Set the entry point
        graph.set_entry_point("reason")
        
        # Compile the graph
        self.graph = graph.compile()

    def reason_about_request(self, state: AgentState):
        """Reason about the user's request and determine what information is needed."""
        messages = state['messages']
        
        # Store the original user query
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        user_query = user_messages[-1].content if user_messages else ""
        
        # Create system message for reasoning
        reasoning_system = """You are a helpful assistant for Budapest tourism and transit information.
Your task is to analyze the user's request and decide:
1. What is the user asking for?
2. What tools would be most appropriate to help fulfill this request?
3. What specific information do you need to gather?

Respond in the following format:
```reasoning
[Your detailed reasoning about what the user is asking and what information you need to gather]
```

```decision
[Either "use_tools" if you need to use tools to fulfill this request, or "respond_directly" if you can answer without tools]
```

Do NOT try to answer the user's question yet - just analyze what would be needed to answer it.
"""
        
        # Add system message for reasoning
        reasoning_messages = [SystemMessage(content=reasoning_system)]
        
        # Add regular messages
        for msg in messages:
            reasoning_messages.append(msg)
        
        # Call the model for reasoning
        reasoning_response = self.reason_model.invoke(reasoning_messages)
        
        # Extract reasoning and decision
        reasoning_match = re.search(r'```reasoning\s*(.*?)\s*```', reasoning_response.content, re.DOTALL)
        decision_match = re.search(r'```decision\s*(.*?)\s*```', reasoning_response.content, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No explicit reasoning provided."
        next_step = decision_match.group(1).strip() if decision_match else "use_tools"  # Default to using tools
        
        # If the decision is to respond directly, create a response
        if next_step == "respond_directly":
            # Create direct response prompt
            direct_response_system = self.system + """
Since the user's question can be answered directly without tools, provide a helpful response based on your general knowledge about Budapest.
"""
            # Create messages for direct response
            direct_response_messages = [SystemMessage(content=direct_response_system)]
            direct_response_messages.extend(messages)
            
            # Get direct response
            direct_response = self.model.invoke(direct_response_messages)
            
            # Add direct response to messages
            messages.append(direct_response)
        
        # Return updated state with reasoning and next step
        return {
            'messages': messages if next_step == "respond_directly" else messages,
            'reasoning_output': reasoning,
            'next_step': next_step,
            'tool_history': [],
            'has_used_tools': False,
            'user_query': user_query
        }

    def determine_next_step(self, state: AgentState):
        """Determine whether to use tools or respond directly."""
        return state.get('next_step', 'use_tools')

    def exists_action(self, state: AgentState):
        """Check if the last message contains any tool calls."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(getattr(result, 'tool_calls', [])) > 0

    def should_continue(self, state: AgentState):
        """
        Determine if we should continue with more tool calls or proceed to finalize.
        """
        # Check if the last message is an AI message with tool calls
        last_message = state['messages'][-1]
        has_more_tools = hasattr(last_message, 'tool_calls') and len(getattr(last_message, 'tool_calls', [])) > 0
        
        return has_more_tools

    def call_openai(self, state: AgentState):
        """Call the language model to generate a response or tool calls."""
        messages = state['messages']
        reasoning = state.get('reasoning_output', '')
        
        # Add system message if not present, including the reasoning
        if self.system:
            tool_system = self.system + f"""
I've analyzed the user's request and here's my reasoning:
{reasoning}

Based on this reasoning, I'll now use the appropriate tools to help answer the query.
"""
            # Replace existing system message or add new one
            has_system = False
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage):
                    messages[i] = SystemMessage(content=tool_system)
                    has_system = True
                    break
            
            if not has_system:
                messages = [SystemMessage(content=tool_system)] + messages
        
        # Call the model and get a response
        message = self.model.invoke(messages)
        
        # Return the updated state with the new message
        return {
            'messages': [message],
            'reasoning_output': reasoning,
            'next_step': state.get('next_step'),
            'tool_history': state.get('tool_history', []),
            'has_used_tools': state.get('has_used_tools', False),
            'user_query': state.get('user_query', '')
        }

    def take_action(self, state: AgentState):
        """Execute any tool calls from the language model."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        tool_history = state.get('tool_history', [])
        
        # Process each tool call
        for t in tool_calls:
            if t['name'] not in self.tools:
                result = f"Invalid tool name: {t['name']}. Retry."
            else:
                try:
                    # Call the tool with the arguments
                    result = self.tools[t['name']].invoke(t['args'])
                    
                    # Add to tool history
                    tool_history.append({
                        'tool': t['name'],
                        'args': t['args'],
                        'result': result
                    })
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
                    
            # Create a tool message with the result
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            
        # Return the updated state with the tool results
        return {
            'messages': results,
            'reasoning_output': state.get('reasoning_output'),
            'next_step': state.get('next_step'),
            'tool_history': tool_history,
            'has_used_tools': True,
            'user_query': state.get('user_query', '')
        }

    def finalize_response(self, state: AgentState):
        """Generate a final response based on all the information gathered."""
        messages = state['messages']
        reasoning = state.get('reasoning_output', '')
        tool_history = state.get('tool_history', [])
        has_used_tools = state.get('has_used_tools', False)
        user_query = state.get('user_query', '')
        
        # If we haven't used any tools, the last message should already be the final response
        if not has_used_tools:
            return state
        
        # Create a message to summarize findings for the finalization model
        finalization_prompt = f"""
Your job is to respond to this user query about Budapest:
"{user_query}"

Here's my analysis of what they're asking:
{reasoning}

I've gathered information using these tools:
"""
        
        for i, tool in enumerate(tool_history):
            tool_name = tool['tool']
            args_str = str(tool['args'])
            result_str = str(tool['result'])
            
            # Truncate long content
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
                
            finalization_prompt += f"\n{i+1}. Used {tool_name} with args: {args_str}\n"
            finalization_prompt += f"   Result: {result_str}\n"
        
        finalization_prompt += """
Please create a helpful, informative response that answers their question.

Follow these formatting guidelines:
- For route information, use emojis (🚆, 🚍, 🚇, 🚶) and clear numbering
- For attractions, mention that info comes from web search
- Use the same language the user used in their query
- End with 1-2 relevant follow-up questions

Your response should be comprehensive, well-organized, and user-friendly.
"""
        
        # Create a system message with our original prompt
        final_messages = [
            SystemMessage(content=self.system),
            HumanMessage(content=finalization_prompt)
        ]
        
        # Generate the final response
        final_response = self.finalize_model.invoke(final_messages)
        
        # Return state with the final response added to messages
        return {
            'messages': messages + [final_response],
            'reasoning_output': reasoning,
            'next_step': state.get('next_step'),
            'tool_history': tool_history,
            'has_used_tools': has_used_tools,
            'user_query': user_query
        }

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

# Create the model instance
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Define the tools available to the agent
tools = [
    parse_input_tool, 
    directions_tool, 
    attractions_tool,
    extract_attractions_tool,
    attraction_info_tool
]

# Create the agent instance
budapest_agent = Agent(model, tools, system=prompt)
