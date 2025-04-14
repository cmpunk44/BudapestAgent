import streamlit as st
from langchain_core.messages import HumanMessage
from agent import budapest_agent  # a frissített Agent osztály itt van

# Streamlit oldalbeállítások
st.set_page_config(page_title="Budapest Agent", layout="wide")  # fontos: wide

# Két hasábos elrendezés: Bal (Debug), Jobb (Chat)
left_col, right_col = st.columns([1, 2])  # Debug : Chat

# === 🐞 DEBUG PANEL ===
with left_col:
    st.markdown("### 🐞 Debug Panel")
    history = budapest_agent.get_history()
    if history:
        for i, msg in enumerate(history):
            role = msg.type.upper()
            st.markdown(f"**{i+1}. {role}**")
            try:
                # JSON ha lehet
                parsed = json.loads(msg.content)
                st.json(parsed)
            except:
                st.code(str(msg.content), language="markdown")

        # Az utolsó üzenet tool válasza részletesen
        if history[-1].type == "tool":
            st.markdown("**🔧 Last Tool Output (parsed)**")
            try:
                st.json(json.loads(history[-1].content))
            except:
                st.warning("Tool output is not valid JSON.")

# === 💬 CHAT UI ===
with right_col:
    st.title("🚌 Budapest Tömegközlekedési Asszisztens")
    st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat vagy látványosságokat!")

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
                continue
            else:
                st.markdown(f"**🤖 Asszisztens:** {msg.content}")
