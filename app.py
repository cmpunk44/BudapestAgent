import streamlit as st
from agent import budapest_agent
from langchain_core.messages import HumanMessage

# Ez legyen az első!
st.set_page_config(page_title="Budapest Agent", layout="centered")

# === ChatGPT-stílus: chat üzenetek + bemenet alul ===
st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Chat üzenetek megjelenítése (beszélgetés történet)
history = budapest_agent.get_history()
if history:
    for msg in history:
        if msg.type == "human":
            st.markdown(f"**👤 Te:** {msg.content}")
        elif msg.type == "ai":
            st.markdown(f"**🤖 Asszisztens:** {msg.content}")

# Bemeneti mező
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

# Gombok: Küldés és Törlés
col1, col2 = st.columns([4, 1])
with col1:
    if st.button("Küldés") and user_input:
        with st.spinner("Dolgozom..."):
            try:
                budapest_agent.add_user_message(user_input)
                result = budapest_agent.run()
                response = result["messages"][-1]
                budapest_agent.history.append(response)
            except Exception as e:
                st.error(f"Hiba történt: {str(e)}")
with col2:
    if st.button("🧹 Törlés"):
        budapest_agent.reset_history()
with st.expander("🛠️ Tool Call Debug"):
    history = budapest_agent.get_history()
    tool_calls = [msg for msg in history if msg.type == "tool"]

    if not tool_calls:
        st.info("No tool calls in this conversation yet.")
    else:
        for i, msg in enumerate(tool_calls):
            st.markdown(f"**🔧 Tool #{i+1}: `{msg.name}`**")
            try:
                st.json(json.loads(msg.content))
            except:
                st.code(msg.content)
