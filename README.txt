# Budapest Public Transport and Tourist Recommender Agent

A LangGraph + Streamlit alapú interaktív asszisztens Budapest tömegközlekedéséhez és turisztikai látványosságaihoz.

## 🚀 Funkciók

- Természetes nyelvű útvonaltervezés tömegközlekedéssel (Google Directions API)
- Közeli turisztikai látnivalók ajánlása (Google Places API)
- LangGraph + ReAct ágens használata
- Streamlit alapú böngészőben futó UI

---

## 📦 Telepítés

1. **Klónozd a repót vagy másold a fájlokat:**

```bash
git clone https://github.com/sajat-felhasznalo/budapest-agent-app.git
cd budapest-agent-app
```

2. **Hozz létre egy `.env` fájlt** a `.env.example` alapján:

```
cp .env.example .env
```

Majd szerkeszd ki a saját API kulcsaiddal:

```env
OPENAI_API_KEY=sk-...
MAPS_API_KEY=AIza...
```

3. **Telepítsd a szükséges csomagokat:**

```bash
pip install -r requirements.txt
```

Ha nincs `requirements.txt`, telepítsd kézzel:
```bash
pip install streamlit langchain_openai langchain_core langgraph python-dotenv
```

---

## 🧪 Futtatás lokálisan

```bash
streamlit run app.py
```

Ez megnyitja az alkalmazást a böngészőben. Írd be, hogy honnan hová szeretnél menni, és az asszisztens útvonalat + látnivalókat javasol!

---

## 🛡️ Biztonság

- A `.env` fájl tartalmazza az **API kulcsokat**, ezt **ne töltsd fel GitHubra**!
- A `.gitignore` gondoskodik róla, hogy a `.env` és cache fájlok ne kerüljenek verziókövetésre.

---

## 📄 Fájlstruktúra

```bash
budapest-agent-app/
├── app.py               # Streamlit UI
├── agent.py             # LangGraph alapú agent
├── .env.example         # API kulcs sablon
├── .gitignore           # Git kizárások
└── README.md            # Dokumentáció
```

---

## 🔮 Tervek

- Webes keresés tool
- Térképes vizualizáció


---

## 🧠 Szerző

Szalay Miklós Márton  
Adattudomány MSc, Pannon Egyetem

Témavezető: Dr. Dulai Tibor
