import streamlit as st
import json
import re
from agent import budapest_agent
from langchain_core.messages import ToolMessage

# === CSS stílusok (rögzített input + scrollozható chat) ===
st.markdown("""
    <style>
    .chat-container {
        max-height: 75vh;
        overflow-y: auto;
        padding-right: 10px;
    }
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-bubble {
        background-color: #f1f1f1;
    }
    .ai-bubble {
        background-color: #e6f4ea;
    }
    .fixed-input {
        position: fixed;
        bottom: 0;
        width: 65%;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 999;
    }
    </style>
""", unsafe_allow_html=True)

# === Oldalelrendezés ===
st.set_page_config(page_title="Budapest Agent", layout="wide")
left_col, right_col = st.columns([1, 2])

# === 🐞 BAL PANEL: DEBUG TOOL INFO ===
with left_col:
    st.markdown("### 🐞 Debug Panel")
    history = budapest_agent.get_history()
    if history:
        for i, msg in enumerate(history):
            if msg.type == "tool":
                st.markdown(f"**🔧 Tool Called:** `{msg.name}`")
                try:
                    parsed = json.loads(msg.content)
                    st.json(parsed)
                except:
                    st.code(msg.content)
            elif msg.type == "ai":
                # LLM szöveg, keresünk Thought/Action-t is ha van
                st.markdown(f"**🤖 LLM output (step {i+1}):**")
                st.code(msg.content[:1000] + ("..." if len(msg.content) > 1000 else ""))

# === 💬 JOBB PANEL: CHATGPT-STYLE CHAT ===
with right_col:
    st.title("🚌 Budapest Asszisztens")

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in budapest_agent.get_history():
        if msg.type == "human":
            st.markdown(f'<div class="chat-bubble user-bubble"><strong>👤 Te:</strong><br>{msg.content}</div>', unsafe_allow_html=True)
        elif msg.type == "ai":
            st.markdown(f'<div class="chat-bubble ai-bubble"><strong>🤖 Asszisztens:</strong><br>{msg.content}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # === FIX INPUT DOBOZ ===
    with st.container():
        st.markdown('<div class="fixed-input">', unsafe_allow_html=True)

        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Írd be a kérdésed", key="chat_input", label_visibility="collapsed",
                                       placeholder="Pl. Hogyan jutok el az ipar utcától a hősök teréig?")
        with col2:
            if st.button("Küldés"):
                if user_input:
                    with st.spinner("Dolgozom..."):
                        try:
                            budapest_agent.add_user_message(user_input)
                            result = budapest_agent.run()
                            budapest_agent.history.append(result["messages"][-1])
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Hiba történt: {str(e)}")

        if st.button("🧹 Új beszélgetés"):
            budapest_agent.reset_history()
            st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)
