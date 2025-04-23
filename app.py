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
from agent import budapest_agent, ThoughtMessage

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
        show_thinking = st.toggle("Gondolkodási folyamat mutatása", value=True)
        
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
                if isinstance(message, ThoughtMessage) and show_thinking:
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown(message.content)
                elif not isinstance(message, ThoughtMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
            elif isinstance(message, ToolMessage):
                # Optionally show tool results
                if show_tools:
                    with st.chat_message("system", avatar="🛠️"):
                        st.text(f"Tool: {message.name}")
                        try:
                            # Try to parse as JSON
                            tool_result = json.loads(message.content.replace("'", '"'))
                            st.json(tool_result)
                        except:
                            # Show as plain text otherwise
                            if len(message.content) > 300:
                                st.text(message.content[:300] + "...")
                            else:
                                st.text(message.content)
        
        # User input in the first column
        user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
    
    # Debug panel in second column
    with cols[1]:
        st.title("🔍 Developer Mode")
        st.markdown("### Agent Process")
        
        if st.session_state.debug_info:
            for i, interaction in enumerate(st.session_state.debug_info):
                with st.expander(f"Query {i+1}: {interaction['user_query'][:30]}...", expanded=(i == len(st.session_state.debug_info)-1)):
                    # Display thoughts if available
                    if "thoughts" in interaction and interaction["thoughts"]:
                        st.markdown("### 🧠 Thinking Steps")
                        for j, thought in enumerate(interaction["thoughts"]):
                            with st.expander(f"Thought {j+1}"):
                                st.json(thought)
                        st.markdown("---")
                    
                    # Display tool calls
                    for step in interaction.get('steps', []):
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
            if isinstance(message, ThoughtMessage) and show_thinking:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(message.content)
            elif not isinstance(message, ThoughtMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        elif isinstance(message, ToolMessage) and show_tools:
            with st.chat_message("system", avatar="🛠️"):
                st.text(f"Tool: {message.name}")
                try:
                    # Try to parse as JSON
                    tool_result = json.loads(message.content.replace("'", '"'))
                    st.json(tool_result)
                except:
                    # Show as plain text otherwise
                    if len(message.content) > 300:
                        st.text(message.content[:300] + "...")
                    else:
                        st.text(message.content)
    
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
    
    # We need to rerun the app to show the updated messages
    st.rerun()

# If we have messages and the last one is from the user, generate a response
if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
    # Get response from agent
    with st.spinner("Gondolkodom..."):
        # Use the last user message
        agent_input = st.session_state.messages[-1]
        
        # Get all previous messages for context (exclude thought messages)
        previous_messages = []
        for msg in st.session_state.messages:
            if not isinstance(msg, ThoughtMessage):
                previous_messages.append(msg)
        
        try:
            # Initialize data structures for this interaction
            current_debug_info = {
                "user_query": agent_input.content,
                "steps": [],
                "thoughts": []
            }
            
            # Execute the agent 
            result = budapest_agent.graph.invoke(
                {
                    "messages": previous_messages, 
                    "thoughts": [], 
                    "current_thought": None
                },
                {"recursion_limit": 15}
            )
            
            # Process all messages
            for message in result.get("messages", []):
                # For thought messages, add to debug and optionally display
                if isinstance(message, ThoughtMessage):
                    # Add to debug info
                    if message.thinking:
                        current_debug_info["thoughts"].append(message.thinking)
                    
                    # Add to chat history
                    st.session_state.messages.append(message)
                    
                # For tool messages, add to debug and optionally display
                elif isinstance(message, ToolMessage):
                    # Add to debug info
                    current_debug_info["steps"].append({
                        "tool": message.name,
                        "result": message.content,
                        "step": "tool_result"
                    })
                    
                    # Add to chat history
                    st.session_state.messages.append(message)
                    
                # For normal AI messages, add to chat
                elif isinstance(message, AIMessage):
                    st.session_state.messages.append(message)
            
            # Add the debug info to the session state
            st.session_state.debug_info.append(current_debug_info)
            
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")
            st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))
        
        # We need to rerun to display all the changes
        st.rerun()

# Add footer
st.markdown("---")
cols = st.columns(3)
with cols[1]:
    st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
