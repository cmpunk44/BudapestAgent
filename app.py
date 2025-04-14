import streamlit as st
import json
import re
from langchain_core.messages import HumanMessage
from agent import budapest_agent  # biztosan a friss agent.py-re hivatkozz

# Oldalbeállítás
st.set_page_config(page_title="Budapest Agent", layout="wide")

# Két oszlopos elrendezés: bal = Debug, jobb = Chat
left_col, right_col = st.columns([1, 2])

# === 🐞 DEBUG PANEL ===
with left_col:
    st.markdown("### 🐞 Debug Panel")

    history = budapest_agent.get_history()
    if history:
        for i, msg in enumerate(history):
            if msg.type != "ai":
                continue

            st.markdown(f"**🧠 Step {i+1} – LLM Reasoning**")
            content = msg.content

            # Regex: ReAct-style blokkok kinyerése
            thought = re.search(r"Thought:\s*(.*)", content)
            action = re.search(r"Action:\s*(.*)", content)
            action_input = re.search(r"Action Input:\s*(.*)", content)
            observation = re.search(r"Observation:\s*(.*)", content)
            final_answer = re.search(r"Final Answer:\s*(.*)", content)

            if thought:
                st.markdown(f"- **Thought:** {thought.group(1)}")
            if action:
                st.markdown(f"- **Action:** `{action.group(1)}`")
            if action_input:
                st.markdown(f"- **Action Input:** `{action_input.group(1)}`")
            if observation:
                st.markdown(f"- **Observation:** {observation.group(1)}")
            if final_answer:
                st.markdown(f"- **✅ Final Answer:** {final_answer.group(1)}")

            if not any([thought, action, action_input, observation, final_answer]):
                st.code(content, language="markdown")

        # ToolMessage tartalom külön megjelenítése
        last = history[-1]
        if last.type == "tool":
            st.markdown("**🔧 Tool Response (parsed)**")
            try:
                st.json(json.loads(last.content))
            except:
                st.warning("Tool output is not valid JSON.")

# === 💬 CHAT UI ===
with right_col:
    st.title("🚌 Budapest Tömegközlekedési Asszisztens")
    st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat vagy látnivalókat!")

    user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

    # Új beszélgetés gomb
    if st.button("🧹 Új beszélgetés"):
        budapest_agent.reset_history()

    # Küldés gomb
    if st.button("Küldés") and user_input:
        with st.spinner("Dolgozom a válaszon..."):
            try:
                budapest_agent.add_user_message(user_input)
                result = budapest_agent.run()
                response = result["messages"][-1]
                budapest_agent.history.append(response)
            except Exception as e:
                st.error(f"Hiba történt: {str(e)}")

    # Beszélgetés megjelenítése
    if budapest_agent.get_history():
        st.markdown("### 💬 Beszélgetés")
        for msg in budapest_agent.get_history():
            if msg.type == "human":
                st.markdown(f"**👤 Te:** {msg.content}")
            elif msg.type == "tool":
                continue  # Debugban kezeljük
            else:
                st.markdown(f"**🤖 Asszisztens:** {msg.content}")
