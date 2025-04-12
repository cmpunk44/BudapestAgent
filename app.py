# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from agent import budapest_agent

st.set_page_config(page_title="Budapest Agent", layout="centered")

st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írj be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Inicializáljuk az állapotot
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Gomb oldalpanelben
if st.sidebar.button("🗑️ Törlés / Újrakezdés"):
    st.session_state.chat_history = []
    st.rerun()

# Bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            result = budapest_agent.graph.invoke({"messages": st.session_state.chat_history})
            st.session_state.chat_history.extend(result["messages"])
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")

# Tool-hívások dinamikusan megjelenítve a sidebarban
tool_messages = [msg for msg in reversed(st.session_state.chat_history) if isinstance(msg, ToolMessage)]
if tool_messages:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Tool hívások")
    for msg in tool_messages:
        st.sidebar.markdown(f"**Tool:** `{msg.name}`")
        st.sidebar.code(msg.content, language="json")

# Megjelenítés (fordított sorrend, utolsó üzenet legfelül)
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Beszélgetés")
    for msg in reversed(st.session_state.chat_history):
        role = "👤" if msg.type == "human" else ("🤖" if msg.type == "ai" else None)
        if role:
            st.markdown(f"**{role}** {msg.content}")
