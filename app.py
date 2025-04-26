# app.py
# Simple Streamlit UI for Budapest tourism and transit agent
# Modified to work with the improved assistant → reason → action flow
# Author: Szalay Miklós Márton
# Modified by: Claude 3.7 Sonnet

import streamlit as st

# IMPORTANT: set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="Budapest Explorer",
    page_icon="🇭🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import budapest_agent
from itinerary_agent import create_itinerary  # Import the itinerary function

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

# Initialize session state for active tab
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"

# Initialize session state for itinerary
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None

# Initialize session state for reasoning
if "reasoning" not in st.session_state:
    st.session_state.reasoning = []

# Function to change tabs
def set_tab(tab_name):
    st.session_state.active_tab = tab_name
    
# Simple sidebar with app info
with st.sidebar:
    st.title("Budapest Explorer")
    st.markdown("""
    **Funkciók:**
    - 🚌 Tömegközlekedési útvonaltervezés
    - 🏛️ Látnivalók ajánlása
    - 🍽️ Éttermek, kávézók keresése
    """)
    
    # Add prominent tab buttons at the top of the sidebar
    st.write("## Válassz funkciót / Choose function:")
    
    # Create two columns for the buttons
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
    
    # Settings in an expandable section
    with st.expander("Beállítások / Settings"):
        # Transportation mode selection
        transport_mode = st.selectbox(
            "Közlekedési mód / Transportation mode:",
            ["Tömegközlekedés", "Gyalogos", "Kerékpár", "Autó"],
            index=0
        )
        
        # Map transport mode to API values
        transport_mode_map = {
            "Tömegközlekedés": "transit",
            "Gyalogos": "walking", 
            "Kerékpár": "bicycling",
            "Autó": "driving"
        }
        
        # Debug mode toggle
        debug_mode = st.toggle("Developer Mode", value=False)
        
        # Show reasoning toggle
        show_reasoning = st.toggle("Show Agent Reasoning", value=True)
        
    st.caption("© 2025 Budapest Explorer - Pannon Egyetem")

# Display different content based on active tab
if st.session_state.active_tab == "chat":
    # CHAT TAB
    # Main page title
    st.title("🇭🇺 Budapest Explorer - Chat")
    
    # Define show_tools variable
    show_tools = True
    
    # Layout based on debug mode
    if debug_mode:
        # Split screen into chat and debug panels
        cols = st.columns([2, 1])
        
        # Main chat in first column
        with cols[0]:
            # Display chat history with reasoning when available
            for i, message in enumerate(st.session_state.messages):
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                        
                        # After each user message, we should find a corresponding reasoning
                        # if we have one available (we have at most one reasoning per query)
                        if show_reasoning and i//2 < len(st.session_state.reasoning) and st.session_state.reasoning[i//2]:
                            with st.chat_message("system"):
                                st.markdown("**Reasoning:**")
                                st.markdown(f"<div style='background-color: #f0f7fb; padding: 10px; border-left: 5px solid #3498db; margin-bottom: 10px;'>{st.session_state.reasoning[i//2]}</div>", unsafe_allow_html=True)
                
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
                
                elif isinstance(message, ToolMessage) and show_tools:
                    with st.chat_message("system"):
                        st.text(f"Tool: {message.name}")
                        if len(message.content) > 300:
                            st.text(message.content[:300] + "...")
                        else:
                            st.text(message.content)
            
            # User input
            user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
        
        # Debug panel in second column
        with cols[1]:
            st.title("🔍 Developer Mode")
            
            if st.session_state.debug_info:
                for i, interaction in enumerate(st.session_state.debug_info):
                    with st.expander(f"Query {i+1}: {interaction['user_query'][:30]}...", expanded=(i == len(st.session_state.debug_info)-1)):
                        # Display reasoning if available
                        if 'reasoning' in interaction and interaction['reasoning']:
                            st.markdown("### Reasoning")
                            st.markdown(f"<div style='background-color: #f0f7fb; padding: 10px; border-left: 5px solid #3498db; margin-bottom: 10px;'>{interaction['reasoning']}</div>", unsafe_allow_html=True)
                        
                        # Display tool calls
                        if 'steps' in interaction and interaction['steps']:
                            st.markdown("### Tool Calls")
                            for step in interaction['steps']:
                                if step['step'] == 'tool_call':
                                    st.markdown(f"**Tool Called: `{step['tool']}`**")
                                    st.code(json.dumps(step['args'], indent=2), language='json')
                                else:
                                    st.markdown(f"**Tool Result:**")
                                    st.text(step['result'][:500] + ('...' if len(step['result']) > 500 else ''))
                                st.markdown("---")
    else:
        # Simple chat layout without debug panel
        # Display chat history with reasoning when available
        for i, message in enumerate(st.session_state.messages):
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
                    
                    # After each user message, display reasoning if available
                    if show_reasoning and i//2 < len(st.session_state.reasoning) and st.session_state.reasoning[i//2]:
                        with st.chat_message("system"):
                            st.markdown("**Reasoning:**")
                            st.markdown(f"<div style='background-color: #f0f7fb; padding: 10px; border-left: 5px solid #3498db; margin-bottom: 10px;'>{st.session_state.reasoning[i//2]}</div>", unsafe_allow_html=True)
            
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
            
            elif isinstance(message, ToolMessage) and show_tools:
                with st.chat_message("system"):
                    st.text(f"Tool: {message.name}")
                    if len(message.content) > 300:
                        st.text(message.content[:300] + "...")
                    else:
                        st.text(message.content)
        
        # User input
        user_prompt = st.chat_input("Mit szeretnél tudni Budapest közlekedéséről vagy látnivalóiról?")
    
    # Handle user input
    if user_prompt:
        # Add user message to chat history
        user_message = HumanMessage(content=user_prompt)
        st.session_state.messages.append(user_message)
        
        # Add transportation mode context if needed
        if transport_mode != "Tömegközlekedés":
            mode = transport_mode_map[transport_mode]
            context_prompt = f"{user_prompt} (használj {mode} közlekedési módot)"
            agent_input = HumanMessage(content=context_prompt)
        else:
            agent_input = user_message
        
        # Rerun to display the new user message
        st.rerun()
    
    # Process the agent response if there's a pending user message
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        # Show a spinner while processing
        with st.chat_message("assistant"):
            with st.spinner("Gondolkodom..."):
                # Get context from previous messages
                agent_input = st.session_state.messages[-1]
                previous_messages = st.session_state.messages[:-1]
                all_messages = previous_messages + [agent_input]
                
                try:
                    # Track tool usage for debugging
                    current_debug_info = {
                        "user_query": agent_input.content,
                        "steps": []
                    }
                    tool_summary = []
                    
                    # Run the agent with the initial empty state
                    result = budapest_agent.graph.invoke(
                        {
                            "messages": all_messages, 
                            "reasoning": None, 
                            "needs_more_tools": False,
                            "tool_history": [],
                            "user_query": agent_input.content
                        },
                        {"recursion_limit": 15}  # Increased recursion limit
                    )
                    
                    # Get the reasoning from the result
                    if "reasoning" in result and result["reasoning"]:
                        current_reasoning = result["reasoning"]
                        st.session_state.reasoning.append(current_reasoning)
                        current_debug_info["reasoning"] = current_reasoning
                    
                    # Extract the final AI message
                    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                    if ai_messages:
                        final_response = ai_messages[-1]
                    else:
                        final_response = AIMessage(content="I'm sorry, but I encountered an issue processing your request.")
                    
                    # Track tool calls for debugging and summary
                    for message in result["messages"]:
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            for tool_call in message.tool_calls:
                                # Add to debug info
                                current_debug_info["steps"].append({
                                    "tool": tool_call["name"],
                                    "args": tool_call["args"],
                                    "step": "tool_call"
                                })
                                
                                # Add to summary for chat display
                                tool_name = tool_call["name"]
                                args = tool_call["args"]
                                
                                # Format differently based on tool
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
                    
                    # Add debug info to session state
                    st.session_state.debug_info.append(current_debug_info)
                    
                    # Display the response with tool summary
                    response_content = final_response.content
                    
                    # If tool summary exists, add it to the response
                    if tool_summary:
                        tool_section = "\n\n---\n### Használt eszközök:\n" + "\n".join(tool_summary)
                        response_with_tools = response_content + tool_section
                        
                        # Display response (reasoning is now shown after the user message)
                        st.write(response_with_tools)
                        
                        # Add to chat history
                        st.session_state.messages.append(AIMessage(content=response_with_tools))
                    else:
                        # Just show the regular response
                        st.write(response_content)
                        st.session_state.messages.append(AIMessage(content=response_content))
                    
                except Exception as e:
                    # Simple error handling
                    import traceback
                    st.error(f"Hiba történt: {str(e)}")
                    st.code(traceback.format_exc())
                    st.session_state.messages.append(AIMessage(content=f"Sajnos hiba történt: {str(e)}"))
                    # Add an empty reasoning for this error response
                    st.session_state.reasoning.append("")
                
                # Rerun to reset UI state
                st.rerun()

else:
    # ITINERARY PLANNER TAB
    st.title("🇭🇺 Budapest Explorer - Útiterv / Itinerary")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Útiterv készítés / Create Itinerary")
        
        # Itinerary form
        with st.form("itinerary_form"):
            # Starting location
            start_location = st.text_input(
                "Kiindulási pont / Starting location:",
                value="Deák Ferenc tér"
            )
            
            # Available time
            available_time = st.slider(
                "Rendelkezésre álló idő (óra) / Available time (hours):",
                min_value=2,
                max_value=12,
                value=4,
                step=1
            )
            
            # Interests (multiselect)
            interests = st.multiselect(
                "Érdeklődési körök / Interests:",
                options=[
                    "Múzeumok / Museums",
                    "Történelem / History",
                    "Építészet / Architecture",
                    "Gasztronómia / Food",
                    "Természet / Nature",
                    "Vásárlás / Shopping",
                    "Művészet / Art",
                    "Éjszakai élet / Nightlife"
                ],
                default=["Történelem / History", "Építészet / Architecture"]
            )
            
            # Map the selected interests to English for processing
            interest_map = {
                "Múzeumok / Museums": "museums",
                "Történelem / History": "history",
                "Építészet / Architecture": "architecture",
                "Gasztronómia / Food": "food",
                "Természet / Nature": "nature",
                "Vásárlás / Shopping": "shopping",
                "Művészet / Art": "art",
                "Éjszakai élet / Nightlife": "nightlife"
            }
            
            # Transportation mode
            itinerary_transport = st.selectbox(
                "Közlekedési mód / Transportation mode:",
                options=[
                    "Tömegközlekedés / Transit",
                    "Gyalogos / Walking",
                    "Kerékpár / Bicycling",
                    "Autó / Car"
                ],
                index=0
            )
            
            # Map the transport mode
            transport_map = {
                "Tömegközlekedés / Transit": "transit",
                "Gyalogos / Walking": "walking",
                "Kerékpár / Bicycling": "bicycling",
                "Autó / Car": "driving"
            }
            
            # Special requests
            special_requests = st.text_area(
                "Egyéb kívánságok / Special requests:",
                placeholder="Pl.: Szeretnék látni a Parlamentet... / E.g.: I'd like to see the Parliament..."
            )
            
            # Submit button
            submit_button = st.form_submit_button("Útiterv készítése / Create Itinerary")
            
            if submit_button:
                # Show spinner during processing
                with st.spinner("Útiterv készítése folyamatban... / Creating itinerary..."):
                    # Prepare preferences
                    preferences = {
                        "start_location": start_location,
                        "available_time": available_time,
                        "interests": [interest_map[i] for i in interests],
                        "transport_mode": transport_map[itinerary_transport],
                        "special_requests": special_requests
                    }
                    
                    # Call the itinerary function
                    try:
                        itinerary = create_itinerary(preferences)
                        st.session_state.itinerary = itinerary
                    except Exception as e:
                        st.error(f"Hiba történt: {str(e)}")
                        st.session_state.itinerary = "Sajnos hiba történt az útiterv készítése során."
    
    with col2:
        # Display the itinerary if available
        if st.session_state.itinerary:
            st.subheader("Az útiterved / Your Itinerary")
            st.markdown(st.session_state.itinerary)
        else:
            # Show instructions or sample itinerary
            st.info("Töltsd ki az űrlapot az útiterv elkészítéséhez! / Fill out the form to create your itinerary!")
            
            with st.expander("Minta útiterv / Sample Itinerary"):
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

# Simple footer
st.markdown("---")
st.caption("Fejlesztette: Szalay Miklós Márton | Pannon Egyetem")
