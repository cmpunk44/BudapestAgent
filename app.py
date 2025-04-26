# app.py
# Simple Streamlit UI for Budapest tourism and transit agent
# Author: Szalay Miklós Márton
# Modified to include itinerary planner and reasoning visualization
# Thesis project for Pannon University

import streamlit as st

# IMPORTANT: set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="Budapest Explorer",
    page_icon="🇭🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import budapest_agent
from itinerary_agent import create_itinerary  # Import the itinerary function

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# Add reasoning to session state
if "reasoning" not in st.session_state:
    st.session_state.reasoning = []

# Initialize session state for active tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

# Initialize session state for itinerary
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# Function to change tabs
def set_tab(tab_name):
    st.session_state.active_tab = tab_name
    
# Simple sidebar with app info
with st.sidebar:
    st.title("Budapest Explorer")
    st.markdown("""
    **Funkciók:**
    - 🚌 Tömegközlekedési útvonaltervezés
    - 🏛️ Látnivalók ajánlása
    - 🍽️ Éttermek, kávézók keresése
    """)
    
    # Add prominent tab buttons at the top of the sidebar
    st.write("## Válassz funkciót / Choose function:")
    
    # Create two columns for the buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Chat", use_container_width=True, 
                    type="primary" if st.session_state.active_tab == "chat" else "secondary"):
            set_tab("chat")
            st.rerun()
            
    with col2:
        if st.button("📅 Útiterv", use_container_width=True,
                    type="primary" if st.session_state.active_tab == "itinerary" else "secondary"):
            set_tab("itinerary")
            st.rerun()
    
    st.markdown("---")
    
    # Settings in an expandable section
    with st.expander("Beállítások / Settings"):
        # Transportation mode selection
        transport_mode = st.selectbox(
            "Közlekedési mód / Transportation mode:",
            ["Tömegközlekedés", "Gyalogos", "Kerékpár", "Autó"],
            index=0
        )
        
        # Map transport mode to API values
        transport_mode_map = {
            "Tömegközlekedés": "transit",
            "Gyalogos": "walking", 
            "Kerékpár": "bicycling",
            "Autó": "driving"
        }
        
        # Debug mode toggle
        debug_mode = st.toggle("Developer Mode", value=False)
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Display different content based on active tab
if st.session_state.active_tab == "chat":
    # CHAT TAB
    # Main page title
    st.title("🇭🇺 Budapest Explorer - Chat")
    
    # Define show_tools variable (always True now that toggle is removed)
    show_tools = True
    
    # Layout based on debug mode
    if debug_mode:
        # Split screen into chat and debug panels
        cols = st.columns([2, 1])
        
        # Main chat in first column
        with cols[0]:
            # Display chat history
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
                elif isinstance(message, ToolMessage) and show_tools:
                    with st.chat_message("system"):
                        st.text(f"Tool: {message.name}")
                        if len(message.content) > 300:
                            st.text(message.content[:300] + "...")
                        else:
                            st.text(message.content)
            
            # User input
            user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
        
        # Debug panel in second column
        with cols[1]:
            st.title("🔍 Developer Mode")
            
            # Add a section for reasoning
            if st.session_state.reasoning:
                with st.expander("💡 Reasoning", expanded=True):
                    # Display the latest reasoning
                    if st.session_state.reasoning:
                        st.markdown("### Current Reasoning Process:")
                        st.markdown(st.session_state.reasoning[-1])
            
            if st.session_state.debug_info:
                for i, interaction in enumerate(st.session_state.debug_info):
                    with st.expander(f"Query {i+1}: {interaction['user_query'][:30]}...", expanded=(i == len(st.session_state.debug_info)-1)):
                        # Display reasoning if available
                        if i < len(st.session_state.reasoning):
                            st.markdown("### Reasoning:")
                            st.markdown(st.session_state.reasoning[i])
                            st.markdown("---")
                        
                        # Display tool calls
                        for step in interaction['steps']:
                            if step['step'] == 'tool_call':
                                st.markdown(f"**Tool Called: `{step['tool']}`**")
                                st.code(json.dumps(step['args'], indent=2), language='json')
                            else:
                                st.markdown(f"**Tool Result:**")
                                st.text(step['result'][:500] + ('...' if len(step['result']) > 500 else ''))
                            st.markdown("---")
    else:
        # Simple chat layout without debug panel
        # Display chat history
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
            elif isinstance(message, ToolMessage) and show_tools:
                with st.chat_message("system"):
                    st.text(f"Tool: {message.name}")
                    if len(message.content) > 300:
                        st.text(message.content[:300] + "...")
                    else:
                        st.text(message.content)
        
        # User input
        user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
    
    # Handle user input
    if user_prompt:
        # Add user message to chat history
        user_message = HumanMessage(content=user_prompt)
        st.session_state.messages.append(user_message)
        
        # Add transportation mode context if needed
        if transport_mode != "Tömegközlekedés":
            mode = transport_mode_map[transport_mode]
            context_prompt = f"{user_prompt} (használj {mode} közlekedési módot)"
            agent_input = HumanMessage(content=context_prompt)
        else:
            agent_input = user_message
        
        # Rerun to display the new user message
        st.rerun()
    
    # Process the agent response if there's a pending user message
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        # Show a spinner while processing
        with st.chat_message("assistant"):
            with st.spinner("Gondolkodom..."):
                # Get context from previous messages
                agent_input = st.session_state.messages[-1]
                previous_messages = st.session_state.messages[:-1]
                all_messages = previous_messages + [agent_input]
                
                try:
                    # Track tool usage for debugging
                    current_debug_info = {
                        "user_query": agent_input.content,
                        "steps": []
                    }
                    tool_summary = []
                    
                    # Run the agent
                    result = budapest_agent.graph.invoke(
                        {"messages": all_messages, "reasoning": ""},
                        {"recursion_limit": 10}
                    )
                    
                    # Get the final response
                    final_response = result["messages"][-1]
                    
                    # Store the reasoning
                    if "reasoning" in result and result["reasoning"]:
                        st.session_state.reasoning.append(result["reasoning"])
                    
                    # Track tool calls for debugging and summary
                    for message in result["messages"]:
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                # Add to debug info
                                current_debug_info["steps"].append({
                                    "tool": tool_call["name"],
                                    "args": tool_call["args"],
                                    "step": "tool_call"
                                })
                                
                                # Add to summary for chat display
                                tool_name = tool_call["name"]
                                args = tool_call["args"]
                                
                                # Format differently based on tool
                                if tool_name == "attraction_info_tool":
                                    if isinstance(args, dict) and 'attractions' in args:
                                        attractions = args['attractions']
                                        tool_summary.append(f"🔍 **Web keresés**: {attractions}")
                                    else:
                                        tool_summary.append(f"🔍 **Web keresés**: {args}")
                                else:
                                    arg_str = str(args)
                                    if len(arg_str) > 50:
                                        arg_str = arg_str[:50] + "..."
                                    tool_summary.append(f"🛠️ **{tool_name}**({arg_str})")
                                
                        elif isinstance(message, ToolMessage):
                            current_debug_info["steps"].append({
                                "tool": message.name,
                                "result": message.content,
                                "step": "tool_result"
                            })
                    
                    # Add debug info to session state
                    st.session_state.debug_info.append(current_debug_info)
                    
                    # Display the response with tool summary
                    response_content = final_response.content
                    
                    # If tool summary exists, add it to the response
                    if tool_summary:
                        tool_section = "\n\n---\n### Használt eszközök:\n" + "\n".join(tool_summary)
                        response_with_tools = response_content + tool_section
                        st.write(response_with_tools)
                        
                        # Add to chat history
                        st.session_state.messages.append(AIMessage(content=response_with_tools))
                    else:
                        # Just show the regular response
                        st.write(response_content)
                        st.session_state.messages.append(AIMessage(content=response_content))
                    
                except Exception as e:
                    # Simple error handling
                    st.error(f"Hiba történt: {str(e)}")
                    st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))
                
                # Rerun to reset UI state
                st.rerun()

else:
    # ITINERARY PLANNER TAB
    st.title("🇭🇺 Budapest Explorer - Útiterv / Itinerary")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Útiterv készítés / Create Itinerary")
        
        # Itinerary form
        with st.form("itinerary_form"):
            # Starting location
            start_location = st.text_input(
                "Kiindulási pont / Starting location:",
                value="Deák Ferenc
