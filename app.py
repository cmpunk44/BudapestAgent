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
from agent import budapest_agent, ReasoningMessage

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
        
        # Show tool calls in chat
        show_tools = st.toggle("Eszközhívások mutatása a chatben", value=True)
        
        # Show reasoning
        show_reasoning = st.toggle("Gondolkodási folyamat mutatása", value=True)
        
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
                if isinstance(message, ReasoningMessage) and show_reasoning:
                    with st.chat_message("assistant", avatar="🧠"):
                        st.write(message.content)
                elif not isinstance(message, ReasoningMessage):
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
                    # Display reasonings if available
                    if "reasonings" in interaction and interaction["reasonings"]:
                        st.markdown("### 🧠 Reasoning Processes")
                        for j, reasoning in enumerate(interaction["reasonings"]):
                            with st.expander(f"Reasoning {j+1}: {reasoning.get('type', 'analysis')}"):
                                st.json(reasoning)
                        st.markdown("---")
                    
                    # Display tool calls
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
            if isinstance(message, ReasoningMessage) and show_reasoning:
                with st.chat_message("assistant", avatar="🧠"):
                    st.write(message.content)
            elif not isinstance(message, ReasoningMessage):
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
    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("Gondolkodom..."):
            # Use the last user message
            agent_input = st.session_state.messages[-1]
            
            # Get all previous messages for context
            previous_messages = []
            for msg in st.session_state.messages:
                if not isinstance(msg, ReasoningMessage):  # Filter out reasoning messages
                    previous_messages.append(msg)
            
            try:
                # Initialize data structures for this interaction
                current_debug_info = []
                collected_reasonings = []
                
                # Execute the agent with the new state structure
                result = budapest_agent.graph.invoke(
                    {"messages": previous_messages, "reasonings": [], "current_reasoning": None},
                    {"recursion_limit": 15}
                )
                
                # Extract the final response and all messages
                all_result_messages = result["messages"]
                reasonings = result.get("reasonings", [])
                
                # Identify the non-reasoning messages (tool calls and actual responses)
                non_reasoning_messages = []
                for msg in all_result_messages:
                    if isinstance(msg, ReasoningMessage):
                        # Store reasoning data for debug info
                        collected_reasonings.append(msg.reasoning_data)
                        # Add reasoning to chat if enabled
                        if show_reasoning:
                            st.session_state.messages.append(msg)
                    else:
                        non_reasoning_messages.append(msg)
                        
                        # Process tool calls for debugging
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                current_debug_info.append({
                                    "tool": tool_call["name"],
                                    "args": tool_call["args"],
                                    "step": "tool_call"
                                })
                        elif isinstance(msg, ToolMessage):
                            current_debug_info.append({
                                "tool": msg.name,
                                "result": msg.content,
                                "step": "tool_result"
                            })
                
                # Find the final AI response (last non-reasoning message)
                final_response = None
                for msg in reversed(non_reasoning_messages):
                    if isinstance(msg, AIMessage) and not isinstance(msg, ReasoningMessage):
                        final_response = msg
                        break
                
                # If no final response is found, use the last message
                if not final_response and non_reasoning_messages:
                    final_response = non_reasoning_messages[-1]
                
                # Add the debug info to the session state
                st.session_state.debug_info.append({
                    "user_query": agent_input.content,
                    "steps": current_debug_info,
                    "reasonings": collected_reasonings
                })
                
                # Display the response content
                if final_response:
                    response_content = final_response.content
                    
                    # If show tools is enabled, append the tool summary
                    if show_tools and current_debug_info:
                        # Format tool calls for display
                        tool_summary = []
                        for info in current_debug_info:
                            if info['step'] == 'tool_call':
                                tool_name = info['tool']
                                if tool_name == "attraction_info_tool":
                                    args = info['args']
                                    if isinstance(args, dict) and 'attractions' in args:
                                        attractions = args['attractions']
                                        tool_summary.append(f"🔍 **Web keresés**: {attractions}")
                                    else:
                                        tool_summary.append(f"🔍 **Web keresés**: {args}")
                                elif tool_name in ["reasoning_tool", "next_step_tool"]:
                                    tool_summary.append(f"🧠 **Gondolkodási folyamat**: {tool_name}")
                                else:
                                    tool_summary.append(f"🛠️ **{tool_name}**({str(info['args'])[:50]}...)")
                        
                        if tool_summary:
                            tool_section = "\n\n---\n### Használt eszközök:\n" + "\n".join(tool_summary)
                            response_with_tools = response_content + tool_section
                            st.write(response_with_tools)
                            
                            # Add to chat history
                            st.session_state.messages.append(AIMessage(content=response_with_tools))
                        else:
                            st.write(response_content)
                            st.session_state.messages.append(AIMessage(content=response_content))
                    else:
                        # Just show the regular response
                        st.write(response_content)
                        st.session_state.messages.append(AIMessage(content=response_content))
                else:
                    # No response found
                    st.error("Nem sikerült választ generálni.")
                    st.session_state.messages.append(AIMessage(content="Sajnos nem sikerült választ generálni a kérdésedre."))
                
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
