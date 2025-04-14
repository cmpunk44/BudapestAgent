import streamlit as st
from agent import budapest_agent

st.set_page_config(page_title="Budapest Agent", layout="centered")
st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat!")

user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

# Reset gomb (frissítés nélkül!)
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

# Megjelenítés
if budapest_agent.get_history():
    st.markdown("### Beszélgetés")
    for msg in budapest_agent.get_history():
        if msg.type == "human":
            st.markdown(f"**👤 Te:** {msg.content}")
        elif msg.type == "tool":
            continue  # ToolMessage-eket kihagyjuk
        else:
            st.markdown(f"**🤖 Asszisztens:** {msg.content}")
