import streamlit as st
from agent import budapest_agent

st.set_page_config(page_title="Budapest Agent", layout="centered")
st.title("🚌 Budapest Tömegközlekedési Asszisztens")
st.markdown("Írd be, hova szeretnél menni, és ajánlok útvonalat + látnivalókat!")

# Bemenet
user_input = st.text_input("Kérdésed:", placeholder="Pl. Hogyan jutok el az Ipar utcáról a Hősök terére?")

# Gomb: új beszélgetés
if st.button("🧹 Új beszélgetés"):
    budapest_agent.reset_history()
    st.experimental_rerun()

# Gomb: küldés
if st.button("Küldés") and user_input:
    with st.spinner("Dolgozom a válaszon..."):
        try:
            budapest_agent.add_user_message(user_input)
            result = budapest_agent.run()
            output = result["messages"][-1].content
            budapest_agent.history.append(result["messages"][-1])  # LLM válasz mentése

            st.markdown("### Válasz")
            st.write(output)
        except Exception as e:
            st.error(f"Hiba történt: {str(e)}")

# Előzmények megjelenítése
if budapest_agent.get_history():
    st.markdown("### Beszélgetés előzménye")
    for msg in budapest_agent.get_history():
        role = "👤" if msg.type == "human" else "🤖"
        st.markdown(f"**{role}:** {msg.content}")
