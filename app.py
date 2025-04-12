# app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from agent import budapest_agent

st.set_page_config(page_title="Budapest Agent", layout="centered")

st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írj be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Inicializáljuk az állapotot
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Gomb a beszélgetés törlésére
if st.button("🗑️ Törlés / Újrakezdés"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            # Előző beszélgetések + új üzenet
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            result = budapest_agent.graph.invoke({"messages": st.session_state.chat_history})
            response = result["messages"][-1]
            st.session_state.chat_history.append(response)
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")

# Megjelenítés
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Beszélgetés")
    for msg in st.session_state.chat_history:
        role = "👤" if msg.type == "human" else "🤖"
        st.markdown(f"**{role}** {msg.content}")
