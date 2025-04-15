import streamlit as st
import json
import re
from langchain_core.messages import HumanMessage
from agent import budapest_agent

st.set_page_config(page_title="Budapest Agent", layout="wide")

# Két hasáb: Bal = Debug, Jobb = Chat
left_col, right_col = st.columns([1, 2])

# === BAL OLDAL: 🐞 DEBUG PANEL (REACT-STYLE) ===
with left_col:
    st.markdown("### 🐞 Debug Panel")
    history = budapest_agent.get_history()
    if history:
        for i, msg in enumerate(history):
            if msg.type != "ai":
                continue

            st.markdown(f"**Step {i+1}:**")

            content = msg.content

            thought = re.search(r"Thought:\s*(.*)", content)
            action = re.search(r"Action:\s*(.*)", content)
            action_input = re.search(r"Action Input:\s*(.*)", content)
            observation = re.search(r"Observation:\s*(.*)", content)
            final_answer = re.search(r"Final Answer:\s*(.*)", content)

            if thought:
                st.markdown(f"- 🧠 **Thought:** {thought.group(1)}")
            if action:
                st.markdown(f"- 🔧 **Action:** `{action.group(1)}`")
            if action_input:
                st.markdown(f"- 🎯 **Action Input:** `{action_input.group(1)}`")
            if observation:
                st.markdown(f"- 👀 **Observation:** {observation.group(1)[:500]}...")
            if final_answer:
                st.markdown(f"- ✅ **Final Answer:** {final_answer.group(1)}")

        # Tool message külön JSON-ként
        last = history[-1]
        if last.type == "tool":
            st.markdown("**📦 Tool Output (last):**")
            try:
                st.json(json.loads(last.content))
            except:
                st.code(last.content)

# === JOBB OLDAL: 💬 CHATGPT-STYLE CHAT ===
with right_col:
    st.title("🚌 Budapest Asszisztens")
    st.markdown("Kérdezz bátran tömegközlekedésről vagy látványosságokról!")

    # Chatbox alatt van a beviteli mező
    history = budapest_agent.get_history()
    if history:
        for msg in history:
            if msg.type == "human":
                st.markdown(f"""
                <div style="background-color:#f1f1f1;padding:10px;border-radius:8px;margin:10px 0">
                <strong>👤 Te:</strong><br>{msg.content}
                </div>""", unsafe_allow_html=True)
            elif msg.type == "ai":
                st.markdown(f"""
                <div style="background-color:#e6f4ea;padding:10px;border-radius:8px;margin:10px 0">
                <strong>🤖 Asszisztens:</strong><br>{msg.content}
                </div>""", unsafe_allow_html=True)

    user_input = st.text_input("Írd be a kérdésed", placeholder="Pl. Hogyan jutok el az ipar utcától a hősök teréig?")

    col_send, col_clear = st.columns([2, 1])
    with col_send:
        if st.button("Küldés") and user_input:
            with st.spinner("Dolgozom..."):
                try:
                    budapest_agent.add_user_message(user_input)
                    result = budapest_agent.run()
                    budapest_agent.history.append(result["messages"][-1])
                except Exception as e:
                    st.error(f"Hiba történt: {str(e)}")
    with col_clear:
        if st.button("🧹 Új beszélgetés"):
            budapest_agent.reset_history()
            st.experimental_rerun()
