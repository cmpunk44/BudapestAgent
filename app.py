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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Parliament_Building%2C_Budapest%2C_outside.jpg/1280px-Parliament_Building%2C_Budapest%2C_outside.jpg", use_column_width=True)
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
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Create two columns - main chat and debug panel
if debug_mode:
    chat_col, debug_col = st.columns([2, 1])
else:
    chat_col, debug_col = st.columns([1, 0])

# Main content in the chat column
with chat_col:
    st.title("🇭🇺 Budapest Explorer")

    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # User input
    user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")

    if user_prompt:
        # Add user message to chat history
        user_message = HumanMessage(content=user_prompt)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_prompt)
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Gondolkodom..."):
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
                
                try:
                    # Clear debug info for this new interaction
                    current_debug_info = []
                    
                    # Execute the agent and collect all intermediate steps
                    result = budapest_agent.graph.invoke(
                        {"messages": all_messages},
                        {"recursion_limit": 10}  # Limit recursion to prevent infinite loops
                    )
                    
                    # Extract the final response
                    final_response = result["messages"][-1]
                    
                    # Collect all tool calls from the interaction for debugging
                    for message in result["messages"]:
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                current_debug_info.append({
                                    "tool": tool_call["name"],
                                    "args": tool_call["args"],
                                    "step": "tool_call"
                                })
                        elif isinstance(message, ToolMessage):
                            current_debug_info.append({
                                "tool": message.name,
                                "result": message.content,
                                "step": "tool_result"
                            })
                    
                    # Add the debug info to the session state
                    st.session_state.debug_info.append({
                        "user_query": user_prompt,
                        "steps": current_debug_info
                    })
                    
                    # Display the response
                    st.write(final_response.content)
                    
                    # Add to chat history
                    st.session_state.messages.append(AIMessage(content=final_response.content))
                    
                except Exception as e:
                    st.error(f"Hiba történt: {str(e)}")
                    st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))

# Debug panel in the second column
if debug_mode and debug_col:
    with debug_col:
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

# Add footer
st.markdown("---")
cols = st.columns(3)
with cols[1]:
    st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
