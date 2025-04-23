# app.py
# Simple Streamlit UI for Budapest tourism and transit agent
# Author: Szalay Miklós Márton
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

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# Simple sidebar with app info
with st.sidebar:
    st.title("Budapest Explorer")
    st.markdown("""
    **Funkciók:**
    - 🚌 Tömegközlekedési útvonaltervezés
    - 🏛️ Látnivalók ajánlása
    - 🍽️ Éttermek, kávézók keresése
    
    **Példa kérdések:**
    - "Hogyan juthatok el a Nyugati pályaudvartól a Gellért-hegyig?"
    - "Mutass éttermeket a Váci utca közelében"
    - "Milyen múzeumok vannak a Hősök tere környékén?"
    - "Mesélj a Lánchídról"
    - "Mi az a Halászbástya?"
    """)
    
    # Language selection
    language = st.radio("Nyelv / Language:", ["Magyar", "English"])
    
    # Settings in an expandable section
    with st.expander("Beállítások"):
        # Transportation mode selection
        transport_mode = st.selectbox(
            "Közlekedési mód:",
            ["Tömegközlekedés", "Gyalogos", "Kerékpár", "Autó"],
            index=0
        )
        
        # Debug mode toggle
        debug_mode = st.toggle("Developer Mode", value=False)
        
        # Show tool calls in chat
        show_tools = st.toggle("Eszközhívások mutatása a chatben", value=True)
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Main page title
st.title("🇭🇺 Budapest Explorer")

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
        
        if st.session_state.debug_info:
            for i, interaction in enumerate(st.session_state.debug_info):
                with st.expander(f"Query {i+1}: {interaction['user_query'][:30]}...", expanded=(i == len(st.session_state.debug_info)-1)):
                    # Display query type if available
                    if 'query_type' in interaction:
                        st.markdown(f"**Query Type:** {interaction['query_type']}")
                    
                    # Display destination if available
                    if 'destination' in interaction and interaction['destination']:
                        st.markdown(f"**Destination:** {interaction['destination']}")
                    
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
        mode_map = {
            "Gyalogos": "walking",
            "Kerékpár": "bicycling",
            "Autó": "driving",
            "Tömegközlekedés": "transit"
        }
        context_prompt = f"{user_prompt} (használj {mode_map[transport_mode]} közlekedési módot)"
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
                
                # Run the agent with the updated graph
                result = budapest_agent.graph.invoke(
                    {
                        "messages": all_messages,
                        "query_type": "general",
                        "destination": ""
                    },
                    {"recursion_limit": 12}  # Increased limit to handle followup generation
                )
                
                # Extract query analysis information if available
                if "query_type" in result:
                    current_debug_info["query_type"] = result["query_type"]
                
                if "destination" in result and result["destination"]:
                    current_debug_info["destination"] = result["destination"]
                
                # Get the final response
                final_response = None
                for message in reversed(result["messages"]):
                    if isinstance(message, AIMessage):
                        final_response = message
                        break
                
                # Collect tool calls for debugging and display
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
                            
                            # Don't show the followup_suggestions_tool in the summary
                            if tool_name == "followup_suggestions_tool":
                                continue
                                
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
                
                # Display the response with tool summary if enabled
                if final_response:
                    response_content = final_response.content
                    
                    # If tool summary exists and tools should be shown, add it
                    if show_tools and tool_summary:
                        # Only add tool section if not already present (could be added by followup tool)
                        if "Használt eszközök" not in response_content:
                            tool_section = "\n\n---\n### Használt eszközök:\n" + "\n".join(tool_summary)
                            response_with_tools = response_content + tool_section
                            st.write(response_with_tools)
                            
                            # Add to chat history
                            st.session_state.messages.append(AIMessage(content=response_with_tools))
                        else:
                            # Tools already included in response
                            st.write(response_content)
                            st.session_state.messages.append(AIMessage(content=response_content))
                    else:
                        # Just show the regular response
                        st.write(response_content)
                        st.session_state.messages.append(AIMessage(content=response_content))
                else:
                    # No response found
                    st.error("Nem sikerült választ generálni.")
            except Exception as e:
                # Simple error handling
                st.error(f"Hiba történt: {str(e)}")
                st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))
            
            # Rerun to reset UI state
            st.rerun()

# Simple footer
st.markdown("---")
st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
