# app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from agent import budapest_agent, react_executor

# Streamlit oldalbeállítások
st.set_page_config(page_title="Budapest Agent", layout="centered")

st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Agent kiválasztása
agent_choice = st.selectbox("Válaszd ki az ügynök típust:", ["LangGraph Agent", "ReAct Agent"])

# Felhasználói bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            if agent_choice == "LangGraph Agent":
                initial_message = HumanMessage(content=user_input)
                result = budapest_agent.graph.invoke({"messages": [initial_message]})
                output = result["messages"][-1].content

                st.markdown("### Válasz")
                st.write(output)

                st.markdown("---")
                st.markdown("### 🛠 Debug (LangGraph)")
                for msg in result["messages"]:
                    if msg.type == "ai" and msg.content.strip():
                        st.markdown(f"**Thought:** {msg.content}")
                    elif msg.type == "tool" and hasattr(msg, 'name'):
                        lines = msg.content.split("\n", 1)
                        if len(lines) == 2:
                            call_line, observation = lines
                            st.markdown(f"**Action:** `{msg.name}`")
                            st.code(call_line.strip(), language="json")
                            st.markdown("**Observation:**")
                            st.code(observation.strip(), language="json")
                        else:
                            st.markdown(f"**Action:** `{msg.name}`")
                            st.code(msg.content.strip(), language="json")

            else:
                result = react_executor.invoke({"input": user_input})
                st.markdown("### Válasz (ReAct Agent)")
                st.write(result["output"])

        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")
