# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from agent import budapest_agent

st.set_page_config(page_title="Budapest Agent", layout="centered")

st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írj be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Inicializáljuk az állapotot, ha nem létezik
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
            initial_message = HumanMessage(content=user_input)
            st.session_state.chat_history.append(initial_message)
            result = budapest_agent.graph.invoke({"messages": st.session_state.chat_history})
            st.session_state.chat_history.extend(result["messages"])

            st.markdown("### Válasz")
            st.write(result["messages"][-1].content)
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")

# Tool-hívások dinamikusan megjelenítve a sidebarban
tool_messages = [msg for msg in st.session_state.chat_history if isinstance(msg, ToolMessage)]
if tool_messages:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Tool hívások")
    for msg in tool_messages:
        st.sidebar.markdown(f"**Tool:** `{msg.name}`")
        st.sidebar.code(msg.content, language="json")
