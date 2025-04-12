# app.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from react_budapest_agent import budapest_agent

# Oldalbeállítások
st.set_page_config(page_title="Budapest ReAct Agent", layout="centered")
st.title("🚌 Budapest ReAct Tömegközlekedési Asszisztens")
st.markdown("Írd be, hova szeretnél menni, és figyeld meg, hogyan gondolkodik az asszisztens lépésről lépésre!")

# Felhasználói bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Oktogontól a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            initial_message = HumanMessage(content=user_input)
            result = budapest_agent.graph.invoke({"messages": [initial_message]})
            messages = result["messages"]

            st.markdown("### 💬 Beszélgetés")
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"**🧑 Felhasználó:** {msg.content}")
                elif isinstance(msg, ToolMessage):
                    st.markdown(f"**🛠️ Observation ({msg.name}):** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.markdown(f"**🤖 Asszisztens:** {msg.content}")

            st.markdown("\n---\n")
            st.markdown("### 🟢 Összefoglaló válasz")
            st.success(messages[-1].content)

        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")
