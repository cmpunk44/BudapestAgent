# app.py

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

# Initialize session state for chat history and debugging info
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# Sidebar with app info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Parliament_Building%2C_Budapest%2C_outside.jpg/1280px-Parliament_Building%2C_Budapest%2C_outside.jpg", use_container_width=True)
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
    
    language = st.radio("Nyelv / Language:", ["Magyar", "English"])
    
    with st.expander("Beállítások"):
        transport_mode = st.selectbox(
            "Közlekedési mód:",
            ["Tömegközlekedés", "Gyalogos", "Kerékpár", "Autó"],
            index=0
        )
        
        # Debug mode toggle
        debug_mode = st.toggle("Developer Mode", value=False)
        
        # NEW: Option to show tool calls in chat
        show_tools = st.toggle("Eszközhívások mutatása a chatben", value=True)
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Define layout
# Instead of using conditional columns, we'll use a more reliable approach
st.title("🇭🇺 Budapest Explorer")

# If debug mode is enabled, create columns
if debug_mode:
    # Create two columns with explicit proportion
    cols = st.columns([2, 1])
    
    # Main chat area in first column
    with cols[0]:
        # Display chat messages
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        
        # User input in the first column
        user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
    
    # Debug panel in second column
    with cols[1]:
        st.title("🔍 Developer Mode")
        st.markdown("### ReAct Agent Process")
        
        if st.session_state.debug_info:
            for i, interaction in enumerate(st.session_state.debug_info):
                with st.expander(f"Query {i+1}: {interaction['user_query'][:30]}...", expanded=(i == len(st.session_state.debug_info)-1)):
                    for step in interaction['steps']:
                        if step['step'] == 'tool_call':
                            st.markdown(f"**🔧 Tool Called: `{step['tool']}`**")
                            st.code(json.dumps(step['args'], indent=2), language='json')
                        else:
                            st.markdown(f"**📊 Tool Result:**")
                            try:
                                # Try to format as JSON if possible
                                result_json = json.loads(step['result'].replace("'", '"'))
                                st.json(result_json)
                            except:
                                # Otherwise show as text
                                st.text(step['result'][:500] + ('...' if len(step['result']) > 500 else ''))
                        st.markdown("---")
else:
    # No columns, just use main area for chat
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
    
    # User input in main area
    user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")

# Process user input (outside of the column context to avoid issues)
if user_prompt:
    # Add user message to chat history
    user_message = HumanMessage(content=user_prompt)
    st.session_state.messages.append(user_message)
    
    # Add context about transport mode if selected
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
        
    # Get all previous messages for context
    all_messages = st.session_state.messages[:-1]  # Exclude the most recent user message
    all_messages.append(agent_input)
    
    # We need to rerun the app to show the updated messages
    st.rerun()

# If we have messages and the last one is from the user, generate a response
if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
    # Display user message (should already be visible from the rerun)
    
    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("Gondolkodom..."):
            # Use the last user message
            agent_input = st.session_state.messages[-1]
            
            # Get all previous messages for context (excluding the last user message)
            all_messages = st.session_state.messages
            
            try:
                # Clear debug info for this new interaction
                current_debug_info = []
                
                # Execute the agent and collect all intermediate steps
                result = budapest_agent.graph.invoke(
                    {"messages": all_messages},
                    {"recursion_limit": 15}  # Increased recursion limit to handle more complex flows
                )
                
                # Extract the final response
                final_response = result["messages"][-1]
                
                # Track web search usage specifically
                web_search_used = False
                web_search_attractions = []
                web_search_result = ""
                
                # Collect all tool calls from the interaction for debugging
                tool_summary = []
                for message in result["messages"]:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            # Track if attraction_info_tool was used
                            if tool_name == "attraction_info_tool":
                                web_search_used = True
                                if isinstance(tool_args, dict) and "attractions" in tool_args:
                                    web_search_attractions = tool_args["attractions"]
                            
                            # Add to debug info
                            current_debug_info.append({
                                "tool": tool_name,
                                "args": tool_args,
                                "step": "tool_call"
                            })
                            
                            # Add to tool summary for chat display
                            if isinstance(tool_args, dict):
                                # Format arguments as a nice string
                                args_str = ", ".join([f"{k}: {v}" for k, v in tool_args.items()])
                            else:
                                args_str = str(tool_args)
                            
                            # Special formatting for certain tools
                            if tool_name == "attraction_info_tool":
                                tool_summary.append(f"🔍 **Web keresés**: {tool_args.get('attractions', 'Látnivalók')}")
                            else:
                                tool_summary.append(f"🛠️ **{tool_name}**({args_str})")
                            
                    elif isinstance(message, ToolMessage):
                        # Add to debug info
                        current_debug_info.append({
                            "tool": message.name,
                            "result": message.content,
                            "step": "tool_result"
                        })
                        
                        # Track web search results
                        if message.name == "attraction_info_tool":
                            try:
                                result_data = eval(message.content)
                                if isinstance(result_data, dict) and "source" in result_data and result_data["source"] == "web search":
                                    web_search_result = result_data.get("info", "")
                            except:
                                pass
                
                # Add the debug info to the session state
                st.session_state.debug_info.append({
                    "user_query": agent_input.content,
                    "steps": current_debug_info,
                    "web_search_used": web_search_used
                })
                
                # Display the response
                response_content = final_response.content
                
                # If web search was used, add clear indicator to the response content
                if web_search_used and web_search_result:
                    # Create a highlighted web search section
                    web_search_notice = "\n\n---\n"
                    web_search_notice += "ℹ️ **Web keresés eredménye**:\n"
                    web_search_notice += f"A következő információkat webes keresés segítségével találtam a(z) "
                    web_search_notice += ", ".join([f"**{attraction}**" for attraction in web_search_attractions])
                    web_search_notice += " látnivaló(k)ról:\n\n"
                    web_search_notice += f"```\n{web_search_result}\n```\n"
                    
                    # Prepend this to the response
                    response_with_notice = response_content
                    
                    # Only add the web search notice if it's not already mentioned
                    if "Web keresés eredménye" not in response_content:
                        response_with_notice = web_search_notice + response_content
                    
                    st.write(response_with_notice)
                    # Add to chat history with web search notice
                    st.session_state.messages.append(AIMessage(content=response_with_notice))
                else:
                    # If show tools is enabled, append the tool summary to the response
                    if 'show_tools' in locals() and show_tools and tool_summary:
                        tool_section = "\n\n---\n### Használt eszközök:\n" + "\n".join(tool_summary)
                        response_with_tools = response_content + tool_section
                        st.write(response_with_tools)
                        # Add to chat history with tools
                        st.session_state.messages.append(AIMessage(content=response_with_tools))
                    else:
                        # Just show the regular response
                        st.write(response_content)
                        # Add to chat history without tools
                        st.session_state.messages.append(AIMessage(content=response_content))
                
            except Exception as e:
                st.error(f"Hiba történt: {str(e)}")
                st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))
                
            # We need to rerun to reset the "waiting for response" state
            st.rerun()

# Add footer
st.markdown("---")
cols = st.columns(3)
with cols[1]:
    st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
