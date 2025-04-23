# app.py

import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage
from agent import budapest_agent

# Page configuration
st.set_page_config(
    page_title="Budapest Explorer",
    page_icon="🇭🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with app info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Parliament_Building%2C_Budapest%2C_outside.jpg/1280px-Parliament_Building%2C_Budapest%2C_outside.jpg", use_column_width=True)
    st.title("Budapest Explorer")
    st.markdown("""
    **Funkciók:**
    - 🚌 Tömegközlekedési útvonaltervezés
    - 🏛️ Látnivalók ajánlása
    - 🍽️ Éttermek, kávézók keresése
    - 🌤️ Időjárás információ
    
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
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Main content
st.title("🇭🇺 Budapest Explorer")

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
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
                result = budapest_agent.graph.invoke({"messages": all_messages})
                response = result["messages"][-1]
                
                # Display the response
                st.write(response.content)
                
                # Add to chat history
                st.session_state.messages.append(AIMessage(content=response.content))
                
            except Exception as e:
                st.error(f"Hiba történt: {str(e)}")
                st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))

# Add footer
st.markdown("---")
cols = st.columns(3)
with cols[1]:
    st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
