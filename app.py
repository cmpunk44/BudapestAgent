# app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from agent import budapest_agent, llm  # importáljuk a modellt is a prompt módosításához

# ReAct-style rendszerprompt beállítása
llm.system_message = """
You are a ReAct-style assistant that helps users with Budapest public transportation and tourist attractions.
You follow a Thought → Action → Observation loop:

Example:
Thought: I need to determine where the user wants to go.
Action: parse_input_tool({"text": "Ipar utcától Hősök teréig szeretnék menni"})
Observation: {"from": "Ipar utca", "to": "Hősök tere"}
Thought: Now that I know the locations, I need to get directions.
Action: directions_tool({"from_place": "Ipar utca", "to_place": "Hősök tere"})
Observation: {...}
Thought: I should find tourist attractions at the endpoints.
Action: attractions_tool({"start_lat": 47.47, "start_lng": 19.07, "end_lat": 47.51, "end_lng": 19.08})
Observation: {...}
Final Answer: [here comes the answer to display]

Begin reasoning step-by-step.
"""

# Streamlit oldalbeállítások
st.set_page_config(page_title="Budapest Agent", layout="centered")

st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írj be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Felhasználói bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            initial_message = HumanMessage(content=user_input)
            result = budapest_agent.graph.invoke({"messages": [initial_message]})
            output = result["messages"][-1].content

            st.markdown("### Válasz")
            st.write(output)

            # Thought-Action-Observation Debug panel
            st.markdown("---")
            st.markdown("### 🛠 Lépésenkénti gondolatmenet")
            for msg in result["messages"]:
                if msg.type == "ai" and msg.content.strip():
                    st.markdown(f"**Thought:** {msg.content}")
                elif msg.type == "tool" and hasattr(msg, 'name'):
                    st.markdown(f"**Action:** {msg.name}")
                    st.markdown(f"**Observation:**")
                    st.code(msg.content, language="json")

        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")
