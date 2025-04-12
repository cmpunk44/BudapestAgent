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

# Gombok és tool debug oldalsávban
with st.sidebar:
    if st.button("🗑️ Törlés / Újrakezdés"):
        st.session_state.chat_history = []
        st.rerun()

    # Tool-hívások megjelenítése
    st.markdown("---")
    st.markdown("### 🔧 Tool hívások")
    for msg in reversed(st.session_state.chat_history):
        if isinstance(msg, ToolMessage):
            st.markdown(f"**Tool:** `{msg.name}`")
            st.code(msg.content, language="json")

# Bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            result = budapest_agent.graph.invoke({"messages": st.session_state.chat_history})
            response = result["messages"][-1]
            st.session_state.chat_history.append(response)
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")

# Megjelenítés (fordított sorrend, utolsó üzenet legfelül)
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Beszélgetés")
    for msg in reversed(st.session_state.chat_history):
        role = "👤" if msg.type == "human" else "🤖"
        content = msg.content
        if isinstance(msg, ToolMessage):
            content += f"\n _(meghívott tool: `{msg.name}`)_"
        st.markdown(f"**{role}** {content}")
