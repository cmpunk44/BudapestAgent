# app.py
# Egyszerű Streamlit felhasználói felület budapesti turizmus és közlekedési ágenshez
# Szerző: Szalay Miklós Márton
# Módosítva útiterv tervezővel kiegészítve
# Szakdolgozat projekt a Pannon Egyetem számára

import streamlit as st

# FONTOS: set_page_config KELL lennie az első Streamlit parancsnak
st.set_page_config(
    page_title="Budapest Explorer",
    page_icon="🇭🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import budapest_agent
from itinerary_agent import create_itinerary  # Útiterv funkció importálása

# Munkamenet állapot inicializálása a chat előzményekhez
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# Munkamenet állapot inicializálása az aktív fülhöz
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

# Munkamenet állapot inicializálása az útitervhez
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# Függvény a fülek váltásához
def set_tab(tab_name):
    st.session_state.active_tab = tab_name
    
# Egyszerű oldalsáv az alkalmazás információival
with st.sidebar:
    st.title("Budapest Explorer")
    st.markdown("""
    **Funkciók:**
    - 🚌 Tömegközlekedési útvonaltervezés
    - 🏛️ Látnivalók ajánlása
    - 🍽️ Éttermek, kávézók keresése
    """)
    
    # Kiemelt fül gombok hozzáadása az oldalsáv tetejére
    st.write("## Válassz funkciót:")
    
    # Két oszlop létrehozása a gombokhoz
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Chat", use_container_width=True, 
                    type="primary" if st.session_state.active_tab == "chat" else "secondary"):
            set_tab("chat")
            st.rerun()
            
    with col2:
        if st.button("📅 Útiterv", use_container_width=True,
                    type="primary" if st.session_state.active_tab == "itinerary" else "secondary"):
            set_tab("itinerary")
            st.rerun()
    
    st.markdown("---")
    
    # Beállítások egy kinyitható részben
    with st.expander("Beállítások"):
        # Közlekedési mód kiválasztása
        transport_mode = st.selectbox(
            "Közlekedési mód:",
            ["Tömegközlekedés", "Gyalogos", "Kerékpár", "Autó"],
            index=0
        )
        
        # Közlekedési mód leképezése API értékekre
        transport_mode_map = {
            "Tömegközlekedés": "transit",
            "Gyalogos": "walking", 
            "Kerékpár": "bicycling",
            "Autó": "driving"
        }
        
        # Fejlesztői mód kapcsoló
        debug_mode = st.toggle("Fejlesztői Mód", value=False)
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Különböző tartalom megjelenítése az aktív fül alapján
if st.session_state.active_tab == "chat":
    # CHAT FÜL
    # Főoldal címe
    st.title("🇭🇺 Budapest Explorer - Chat")
    
    # show_tools változó definíciója
    show_tools = True
    
    # Elrendezés a fejlesztői mód alapján
    if debug_mode:
        # Képernyő felosztása chat és hibakeresési panelekre
        cols = st.columns([2, 1])
        
        # Fő chat az első oszlopban
        with cols[0]:
            # Chat előzmények megjelenítése
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
                elif isinstance(message, ToolMessage) and show_tools:
                    with st.chat_message("system"):
                        st.text(f"Eszköz: {message.name}")
                        st.text(message.content[:300] + "..." if len(message.content) > 300 else message.content)
            
            # Felhasználói bemenet
            user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
        
        # Fejlesztői panel a második oszlopban
        with cols[1]:
            st.title("🔍 Fejlesztői Mód")
            
            if st.session_state.debug_info:
                for i, interaction in enumerate(st.session_state.debug_info):
                    with st.expander(f"Kérdés {i+1}: {interaction['user_query'][:30]}...", expanded=(i == len(st.session_state.debug_info)-1)):
                        # Eszközhívások megjelenítése
                        for step in interaction['steps']:
                            if step['step'] == 'tool_call':
                                st.markdown(f"**Eszköz hívás: `{step['tool']}`**")
                                st.code(json.dumps(step['args'], indent=2), language='json')
                            else:
                                st.markdown(f"**Eszköz eredmény:**")
                                st.text(step['result'][:500] + ('...' if len(step['result']) > 500 else ''))
                            st.markdown("---")
    else:
        # Egyszerű chat elrendezés fejlesztői panel nélkül
        # Chat előzmények megjelenítése
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
            elif isinstance(message, ToolMessage) and show_tools:
                with st.chat_message("system"):
                    st.text(f"Eszköz: {message.name}")
                    st.text(message.content[:300] + "..." if len(message.content) > 300 else message.content)
        
        # Felhasználói bemenet
        user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
    
    # Felhasználói bemenet kezelése
    if user_prompt:
        # Felhasználói üzenet hozzáadása a chat előzményekhez
        user_message = HumanMessage(content=user_prompt)
        st.session_state.messages.append(user_message)
        
        # Közlekedési mód kontextus hozzáadása, ha szükséges
        if transport_mode != "Tömegközlekedés":
            mode = transport_mode_map[transport_mode]
            context_prompt = f"{user_prompt} (használj {mode} közlekedési módot)"
            agent_input = HumanMessage(content=context_prompt)
        else:
            agent_input = user_message
        
        # Újratöltés, hogy megjelenjen az új felhasználói üzenet
        st.rerun()
    
    # Ágens válaszának feldolgozása, ha van függőben lévő felhasználói üzenet
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        # Spinner megjelenítése feldolgozás közben
        with st.chat_message("assistant"):
            with st.spinner("Gondolkodom..."):
                # Kontextus kinyerése az előző üzenetekből
                agent_input = st.session_state.messages[-1]
                previous_messages = st.session_state.messages[:-1]
                all_messages = previous_messages + [agent_input]
                
                # Eszközhasználat követése hibakereséshez
                current_debug_info = {
                    "user_query": agent_input.content,
                    "steps": []
                }
                tool_summary = []
                
                # Ágens futtatása
                result = budapest_agent.graph.invoke(
                    {"messages": all_messages},
                    {"recursion_limit": 10}
                )
                
                # Végső válasz kinyerése
                final_response = result["messages"][-1]
                
                # Eszközhívások követése hibakereséshez és összefoglalóhoz
                for message in result["messages"]:
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            # Hibakeresési infóhoz hozzáadás
                            current_debug_info["steps"].append({
                                "tool": tool_call["name"],
                                "args": tool_call["args"],
                                "step": "tool_call"
                            })
                            
                            # Összefoglalóhoz hozzáadás a chat megjelenítéshez
                            tool_name = tool_call["name"]
                            args = tool_call["args"]
                            
                            # Eltérő formázás eszköz alapján
                            if tool_name == "attraction_info_tool":
                                if isinstance(args, dict) and 'attractions' in args:
                                    attractions = args['attractions']
                                    tool_summary.append(f"🔍 **Web keresés**: {attractions}")
                                else:
                                    tool_summary.append(f"🔍 **Web keresés**: {args}")
                            else:
                                arg_str = str(args)
                                if len(arg_str) > 50:
                                    arg_str = arg_str[:50] + "..."
                                tool_summary.append(f"🛠️ **{tool_name}**({arg_str})")
                            
                    elif isinstance(message, ToolMessage):
                        current_debug_info["steps"].append({
                            "tool": message.name,
                            "result": message.content,
                            "step": "tool_result"
                        })
                
                # Hibakeresési infó hozzáadása a munkamenet állapothoz
                st.session_state.debug_info.append(current_debug_info)
                
                # Válasz megjelenítése eszköz összefoglalóval
                response_content = final_response.content
                
                # Ha van eszköz összefoglaló, hozzáadjuk a válaszhoz
                if tool_summary:
                    tool_section = "\n\n---\n### Használt eszközök:\n" + "\n".join(tool_summary)
                    response_with_tools = response_content + tool_section
                    st.write(response_with_tools)
                    
                    # Hozzáadás a chat előzményekhez
                    st.session_state.messages.append(AIMessage(content=response_with_tools))
                else:
                    # Csak a normál válasz megjelenítése
                    st.write(response_content)
                    st.session_state.messages.append(AIMessage(content=response_content))
                
                # Újratöltés a UI állapot visszaállításához
                st.rerun()

else:
    # ÚTITERV TERVEZŐ FÜL
    st.title("🇭🇺 Budapest Explorer - Útiterv")
    
    # Két oszlop létrehozása
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Útiterv készítés")
        
        # Útiterv űrlap
        with st.form("itinerary_form"):
            # Kiindulási hely
            start_location = st.text_input(
                "Kiindulási pont:",
                value="Deák Ferenc tér"
            )
            
            # Rendelkezésre álló idő
            available_time = st.slider(
                "Rendelkezésre álló idő (óra):",
                min_value=2,
                max_value=12,
                value=4,
                step=1
            )
            
            # Érdeklődési körök (többszörös kiválasztás)
            interests = st.multiselect(
                "Érdeklődési körök:",
                options=[
                    "Múzeumok",
                    "Történelem",
                    "Építészet",
                    "Gasztronómia",
                    "Természet",
                    "Vásárlás",
                    "Művészet",
                    "Éjszakai élet"
                ],
                default=["Történelem", "Építészet"]
            )
            
            # Kiválasztott érdeklődési körök leképezése angolra a feldolgozáshoz
            interest_map = {
                "Múzeumok": "museums",
                "Történelem": "history",
                "Építészet": "architecture",
                "Gasztronómia": "food",
                "Természet": "nature",
                "Vásárlás": "shopping",
                "Művészet": "art",
                "Éjszakai élet": "nightlife"
            }
            
            # Közlekedési mód
            itinerary_transport = st.selectbox(
                "Közlekedési mód:",
                options=[
                    "Tömegközlekedés",
                    "Gyalogos",
                    "Kerékpár",
                    "Autó"
                ],
                index=0
            )
            
            # Közlekedési mód leképezése
            transport_map = {
                "Tömegközlekedés": "transit",
                "Gyalogos": "walking",
                "Kerékpár": "bicycling",
                "Autó": "driving"
            }
            
            # Speciális kérések
            special_requests = st.text_area(
                "Egyéb kívánságok:",
                placeholder="Pl.: Szeretnék látni a Parlamentet..."
            )
            
            # Elküldés gomb
            submit_button = st.form_submit_button("Útiterv készítése")
            
            if submit_button:
                # Spinner megjelenítése feldolgozás közben
                with st.spinner("Útiterv készítése folyamatban..."):
                    # Preferenciák előkészítése
                    preferences = {
                        "start_location": start_location,
                        "available_time": available_time,
                        "interests": [interest_map[i] for i in interests],
                        "transport_mode": transport_map[itinerary_transport],
                        "special_requests": special_requests
                    }
                    
                    # Útiterv funkció hívása
                    itinerary = create_itinerary(preferences)
                    st.session_state.itinerary = itinerary
    
    with col2:
        # Útiterv megjelenítése, ha elérhető
        if st.session_state.itinerary:
            st.subheader("Az útiterved")
            st.markdown(st.session_state.itinerary)
        else:
            # Utasítások vagy minta útiterv megjelenítése
            st.info("Töltsd ki az űrlapot az útiterv elkészítéséhez!")
            
            with st.expander("Minta útiterv"):
                st.markdown("""
                # Budapest Felfedezése - Egy Napos Útiterv
                
                ## Reggel 10:00 - Hősök tere
                A Hősök tere Budapest egyik ikonikus látványossága, ahol megcsodálhatod a magyar történelem fontos alakjainak szobrait.
                
                **Időtartam:** 30 perc
                
                ## Reggel 10:30 - Városliget
                Sétálj át a Városligetbe, ahol megtalálod a Vajdahunyad várát és a Széchenyi fürdőt.
                
                **Időtartam:** 1 óra
                
                ## Délelőtt 11:30 - Andrássy út
                Haladj végig az Andrássy úton a belváros felé, útközben megcsodálhatod a gyönyörű épületeket.
                
                **Közlekedés:** M1-es metró, 10 perc
                
                ## Déli 12:30 - Ebéd a Gozsdu udvarban
                Élvezd Budapest gasztronómiai kínálatát a Gozsdu udvar valamelyik éttermében.
                
                **Időtartam:** 1 óra
                
                ## Délután 14:00 - Szent István Bazilika
                Látogasd meg Budapest legnagyobb templomát, ahonnan csodálatos kilátás nyílik a városra.
                
                **Időtartam:** 45 perc
                
                ## Délután 15:00 - Duna-part és Parlament
                Sétálj le a Duna-partra és csodáld meg a magyar Parlamentet kívülről.
                
                **Közlekedés:** Gyalog, 15 perc
                
                ## Délután 16:00 - Lánchíd és Budai vár
                Sétálj át a Lánchídon Budára, majd látogasd meg a Budai várat.
                
                **Időtartam:** 2 óra
                
                Ez csak egy minta útiterv. A te személyre szabott útiterved az érdeklődési köreid és a rendelkezésre álló időd alapján készül el.
                """)

# Egyszerű lábléc
st.markdown("---")
st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
