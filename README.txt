Budapest Agent
Overview
Budapest Explorer is an interactive AI assistant designed to help tourists and locals navigate Budapest's transportation system and discover points of interest. The application combines a language model with Google Maps APIs to provide route planning, attraction recommendations, and detailed information about Budapest's landmarks and venues.
Features

🚌 Public Transportation Routing: Get step-by-step directions between any two points in Budapest using public transit
🏛️ Attraction Information: Learn about Budapest's landmarks with accurate, web-sourced information
🍽️ Restaurant & Café Recommendations: Discover dining options near specific locations
🇭🇺/🇬🇧 Bilingual Support: Full functionality in both Hungarian and English

Technical Architecture
The application consists of two main components:

LangGraph Agent: A graph-based conversational agent that manages tools and conversation flow
Streamlit UI: A web-based interface that provides an accessible user experience

Core Technologies

OpenAI GPT-4o-mini: Powers the assistant's natural language understanding and generation
Google Directions API: Provides real-time routing information
Google Places API: Finds nearby points of interest and venues
LangGraph: Manages the agent's reasoning and tool-calling flow
Streamlit: Creates the interactive web application

Tools
The agent leverages several specialized tools:

parse_input_tool: Extracts locations from natural language
directions_tool: Gets route information between locations
attractions_tool: Finds points of interest near coordinates
extract_attractions_tool: Identifies attractions mentioned in queries
attraction_info_tool: Retrieves web-sourced information about attractions

Installation
Prerequisites

Python 3.9+
OpenAI API key
Google Maps API key

Setup

Clone the repository

Install dependencies

bash pip install -r requirements.txt

Create a .env file with your API keys:

OPENAI_API_KEY=your_openai_key_here
MAPS_API_KEY=your_google_maps_key_here

Run the application:

bash streamlit run app.py
Usage Examples
Route Planning

"Hogyan juthatok el a Nyugati pályaudvartól a Gellért-hegyig?"
"How do I get from Heroes' Square to Buda Castle?"

Attraction Information

"Mesélj a Lánchídról"
"What is the Fisherman's Bastion?"

Finding Venues

"Mutass éttermeket a Váci utca közelében"
"Are there any good cafés near the Parliament?"

Development
The application features a Developer Mode that provides insights into the agent's decision-making process. When enabled, it shows:

Tool calls and their parameters
Tool results
Complete conversation flow

To enable Developer Mode, use the toggle in the sidebar settings.
Future Improvements
Potential areas for enhancement:

Integration with Budapest public transport GTFS data for schedule information
Expanded multilingual support
User location awareness
More detailed venue information and reviews
Interactive map display for routes and attractions
