# app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from agent import budapest_agent

# Streamlit oldalbeállítások
st.set_page_config(page_title="Budapest Agent", layout="centered")

st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Felhasználói bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        initial_message = HumanMessage(content=user_input)
        result = budapest_agent.graph.invoke({"messages": [initial_message]})
        output = result["messages"][-1].content

        st.markdown("### Válasz")
        st.write(output)
