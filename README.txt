Budapest Public Transport and Tourist Recommender Agent
Interactive assistant based on LangGraph + Streamlit for Budapest public transportation and tourist attractions.
🚀 Features

Natural language route planning with public transportation (Google Directions API)
Nearby tourist attraction recommendations (Google Places API)
LangGraph-based ReAct (Reasoning and Acting) agent
Web search capability for accurate tourist information
Personalized itinerary creation with time frames and interest preferences
Streamlit-based browser UI
Developer mode for visualizing reasoning steps and tool calls

🧠 Architecture
The system employs two main agent architectures:

ReAct agent (agent.py): Graph-based agent with three main nodes:

Reasoning: Explicit thinking step that analyzes the user request
LLM: The language model that processes the request and decides on tool usage
Action: Tool execution and result processing


Itinerary agent (itinerary_agent.py): Linear workflow-based agent that generates personalized itineraries:

Identifying attractions based on user preferences
Gathering detailed information through web search
Planning routes between attractions
Generating final itinerary with timeline and descriptions



🛠️ Tools
The system uses five main tools:

parse_input_tool: Extract locations from natural language text
directions_tool: Route planning using Google Directions API
attractions_tool: Search for attractions and places using Google Places API
extract_attractions_tool: Identify attraction names from user queries
attraction_info_tool: Search for detailed information about attractions using web search

📦 Installation

Clone the repo or copy the files:

bashgit clone https://github.com/your-username/budapest-agent-app.git
cd budapest-agent-app

Create a .env file based on .env.example:

bashcp .env.example .env
Then edit it with your own API keys:
envOPENAI_API_KEY=sk-...
MAPS_API_KEY=AIza...

Install the required packages:

bashpip install -r requirements.txt
If requirements.txt is not available, install manually:
bashpip install streamlit langchain_openai langchain_core langgraph python-dotenv requests
🧪 Running Locally
bashstreamlit run app.py
This will open the application in your browser. Type where you want to go, and the assistant will suggest routes + attractions!
💻 User Interface
The application has two main functions:

Chat: Natural language conversation with the agent

Route planning between two locations
Searching for attractions and querying information
Dining and program suggestions


Itinerary: Creating personalized itineraries

Specifying starting point
Selecting areas of interest
Setting time frame and transportation mode
Providing specific requests



Developer Mode
The application has a developer mode that displays detailed information:

Reasoning process: Visualization of the agent's thinking step
Tool calls: Tools used by the agent and their parameters
Tool results: Data returned by the tools

🧮 Technical Details
Agent Implementation
The ReAct agent is implemented using the LangGraph framework:
pythongraph = StateGraph(AgentState)
graph.add_node("reason", self.add_reasoning)  # Reasoning node
graph.add_node("llm", self.call_openai)       # LLM node
graph.add_node("action", self.take_action)    # Action node
graph.add_edge("reason", "llm")
graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
graph.add_edge("action", "llm")
graph.set_entry_point("reason")
Language Models
The system uses three different OpenAI language models:

llm (GPT-4o-mini): General requests and tool usage
reasoning_llm (GPT-4o-mini): Reasoning step with lower temperature
search_llm (GPT-4o-search-preview): Web search capability for up-to-date information

🛡️ Security

The .env file contains API keys, do not upload it to GitHub!
The .gitignore ensures that .env and cache files are not tracked by version control.

📄 File Structure
bashbudapest-agent-app/
├── app.py               # Streamlit UI
├── agent.py             # LangGraph-based ReAct agent
├── itinerary_agent.py   # Itinerary creation component
├── .env.example         # API key template
├── .gitignore           # Git exclusions
├── requirements.txt     # Dependencies
└── README.md            # Documentation
🔮 Future Plans

Expanding web search to discover current events
Integrating map visualization
Offline mode for limited internet access
Multilingual tourist recommendations
Performance optimizations to reduce response time


Budapest tömegközlekedési és turisztikai ajánló ágens
LangGraph + Streamlit alapú interaktív asszisztens Budapest tömegközlekedéséhez és turisztikai látványosságaihoz.
🚀 Funkciók

Természetes nyelvű útvonaltervezés tömegközlekedéssel (Google Directions API)
Közeli turisztikai látnivalók ajánlása (Google Places API)
LangGraph alapú ReAct (Reasoning and Acting) ágens használata
Web keresés képesség a turisztikai információk pontosításához
Személyre szabott útiterv készítés időkerettel és érdeklődési körökkel
Streamlit alapú böngészőben futó UI
Developer mód a reasoning lépések és eszközhívások vizualizálásához

🧠 Architektúra
A rendszer két fő ágens architektúrát alkalmaz:

ReAct ágens (agent.py): Gráfalapú ágens három fő csomóponttal:

Reasoning: Explicit gondolkodási lépés, amely elemzi a felhasználói kérést
LLM: A nyelvi modell, amely feldolgozza a kérést és dönt az eszközök használatáról
Action: Az eszközök végrehajtása és az eredmények feldolgozása


Útiterv ágens (itinerary_agent.py): Lineáris workflow-alapú ágens, amely személyre szabott útiterveket generál:

Látványosságok azonosítása a felhasználói preferenciák alapján
Részletes információk gyűjtése web kereséssel
Útvonalak tervezése a látványosságok között
Végső útiterv generálása időbeosztással és leírásokkal



🛠️ Eszközök
A rendszer öt fő eszközt használ:

parse_input_tool: Helyszínek kinyerése természetes nyelvű szövegből
directions_tool: Útvonaltervezés a Google Directions API segítségével
attractions_tool: Látnivalók és helyek keresése a Google Places API segítségével
extract_attractions_tool: Látványosságnevek azonosítása a felhasználói kérésből
attraction_info_tool: Részletes információk keresése a látványosságokról web kereséssel

📦 Telepítés

Klónozd a repót vagy másold a fájlokat:

bashgit clone https://github.com/your-username/budapest-agent-app.git
cd budapest-agent-app

Hozz létre egy .env fájlt a .env.example alapján:

bashcp .env.example .env
Majd szerkeszd ki a saját API kulcsaiddal:
envOPENAI_API_KEY=sk-...
MAPS_API_KEY=AIza...

Telepítsd a szükséges csomagokat:

bashpip install -r requirements.txt
Ha nincs requirements.txt, telepítsd kézzel:
bashpip install streamlit langchain_openai langchain_core langgraph python-dotenv requests
🧪 Futtatás lokálisan
bashstreamlit run app.py
Ez megnyitja az alkalmazást a böngészőben. Írd be, hogy honnan hová szeretnél menni, és az asszisztens útvonalat + látnivalókat javasol!
💻 Felhasználói felület
Az alkalmazás két fő funkcióval rendelkezik:

Chat: Természetes nyelvű beszélgetés az ágenssel

Útvonaltervezés két helyszín között
Látnivalók keresése és információk lekérdezése
Étkezési és programjavaslatok


Útiterv: Személyre szabott útitervek készítése

Kiindulási pont megadása
Érdeklődési körök kiválasztása
Időkeret és közlekedési mód beállítása
Specifikus kérések megadása



Developer mód
Az alkalmazás rendelkezik egy fejlesztői móddal, amely részletes információkat jelenít meg:

Reasoning folyamat: Az ágens gondolkodási lépésének megjelenítése
Eszközhívások: Az ágens által használt eszközök és paramétereik
Eszközeredmények: Az eszközök által visszaadott adatok

🧮 Technikai részletek
Ágensmegvalósítás
A ReAct ágens a LangGraph keretrendszer segítségével van implementálva:
pythongraph = StateGraph(AgentState)
graph.add_node("reason", self.add_reasoning)  # Reasoning csomópont
graph.add_node("llm", self.call_openai)       # LLM csomópont
graph.add_node("action", self.take_action)    # Action csomópont
graph.add_edge("reason", "llm")
graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
graph.add_edge("action", "llm")
graph.set_entry_point("reason")
Nyelvi modellek
A rendszer három különböző OpenAI nyelvi modellt használ:

llm (GPT-4o-mini): Általános kérések és eszközhasználat
reasoning_llm (GPT-4o-mini): Reasoning lépés alacsonyabb hőmérséklettel
search_llm (GPT-4o-search-preview): Web keresési képességgel a naprakész információkhoz

🛡️ Biztonság

A .env fájl tartalmazza az API kulcsokat, ezt ne töltsd fel GitHubra!
A .gitignore gondoskodik róla, hogy a .env és cache fájlok ne kerüljenek verziókövetésre.

📄 Fájlstruktúra
bashbudapest-agent-app/
├── app.py               # Streamlit UI
├── agent.py             # LangGraph alapú ReAct agent
├── itinerary_agent.py   # Útiterv készítő komponens
├── .env.example         # API kulcs sablon
├── .gitignore           # Git kizárások
├── requirements.txt     # Függőségek
└── README.md            # Dokumentáció
🔮 Tervek

Webes keresés bővítése aktuális események felderítéséhez
Térképes vizualizáció integrálása
Offline mód a korlátozott internet hozzáférés esetére
Többnyelvű turisztikai ajánlások
Teljesítményoptimalizálások a válaszidő csökkentésére


🧠 Szerző / Author
Szalay Miklós Márton
