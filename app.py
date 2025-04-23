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
        context_prompt = f
