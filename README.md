# AI Travel Planner 🌍✈️

> **Status: Phase 7 – Travel Assistant Agent** | Destination research, geographic itinerary generation, deterministic budgeting, and source-aware travel assistance are implemented. Other agents and graph orchestration remain planned.

---

## Overview

The **AI Travel Planner** is a production-quality, multi-agent AI application that collaborates through a network of specialized AI agents to generate personalized, end-to-end travel plans. Each agent handles a distinct domain — research, itinerary design, budget estimation, weather intelligence, and plan validation — before a final synthesizer compiles everything into a coherent travel guide.

---

## Problem Statement

Planning a trip is time-consuming and fragmented. Travellers must research destinations, compare costs, check weather windows, plan day-by-day itineraries, and validate logistics — often across dozens of tabs. The AI Travel Planner solves this by orchestrating multiple specialized AI agents that perform each of these tasks autonomously and collaboratively, delivering a comprehensive travel plan in a single interaction.

---

## Project Objectives

- Build a modular, extensible multi-agent system using **LangGraph**.
- Provide personalized travel recommendations powered by **Google Gemini**.
- Support real-time web research via **Tavily** and weather data via **Open-Meteo**.
- Expose a clean **FastAPI** backend for programmatic access.
- Deliver a conversational **Streamlit** frontend for end-users.
- Persist user preferences and plans in **SQLite**.
- Enable full observability through optional **LangSmith** tracing.

---

## Planned Multi-Agent Architecture

```
User Input
    │
    ▼
┌─────────────────────┐
│  Coordinator Agent  │  ← Understands intent, routes tasks
└──────────┬──────────┘
           │
     ┌─────┼─────────────────────────┐
     ▼     ▼                         ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────┐
│ Destination  │  │   Itinerary   │  │  Budget Planning │
│  Research    │  │   Planning    │  │     Agent        │
│    Agent     │  │    Agent      │  └──────────────────┘
└──────────────┘  └───────────────┘
                                      ┌──────────────────┐
                                      │  Travel Assistant│
                                      │  Agent (Weather, │
                                      │  Transport, Tips)│
                                      └──────────────────┘
     All agents feed into:

     ┌──────────────────┐     ┌──────────────────────┐
     │ Validation Agent │────▶│  Response Generator  │
     └──────────────────┘     └──────────────────────┘
                                        │
                                        ▼
                               Final Travel Plan
```

| Agent | Responsibility |
|---|---|
| **Coordinator** | Parses user intent; dispatches tasks to specialist agents |
| **Destination Research** | Retrieves attractions, local culture, best seasons via Tavily |
| **Itinerary Planner** | Generates structured day-by-day itineraries |
| **Budget Planner** | Estimates flights, hotels, meals, activities; optimizes spend |
| **Travel Assistant** | Fetches weather, transportation options, packing tips |
| **Validation** | Checks plan consistency, feasibility, and completeness |
| **Response Generator** | Merges all outputs into a polished final travel guide |

All agents share a **typed state graph** managed by LangGraph and communicate through a well-defined shared state object.

---

## Technology Stack

| Layer | Technology |
|---|---|
| AI Orchestration | LangGraph |
| LLM Framework | LangChain + LangChain Google GenAI |
| LLM Provider | Google Gemini |
| Web Search | Tavily |
| Weather Data | Open-Meteo |
| Location / Places | Geoapify (future) |
| Tracing & Debugging | LangSmith |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data Validation | Pydantic v2 |
| Database | SQLite (via SQLAlchemy) |
| Configuration | python-dotenv |
| Testing | pytest + pytest-asyncio |

---

## Project Structure

```
AI-Travel-Planner/
│
├── backend/
│   ├── agents/          # Research, itinerary, budget, and travel assistant agents
│   ├── graph/           # Shared LangGraph state (Phase 3); workflow planned
│   ├── tools/           # Tavily search, route optimization, and Open-Meteo weather
│   ├── models/          # Pydantic data models and shared state
│   ├── api/             # FastAPI routes and middleware
│   ├── database/        # SQLAlchemy models and DB session management
│   ├── config.py        # Centralised configuration and env-var loading
│   ├── llm.py           # Reusable Gemini LLM initialisation (Phase 2)
│   └── main.py          # FastAPI application entry point
│
├── scripts/
│   └── test_gemini.py   # Interactive Gemini test script
│
├── frontend/            # Streamlit UI (Phase 4+)
│
├── tests/
│   └── test_gemini.py   # Gemini unit + integration tests
│
├── docs/                # Project documentation
│
├── verify_setup.py      # Phase 1 setup verification script
├── .env.example         # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- `pip` and `venv`

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/AI-Travel-Planner.git
cd AI-Travel-Planner
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Variable Setup

```bash
# Copy the template
cp .env.example .env
```

Open `.env` and fill in your actual API keys:

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Google Gemini API key |
| `GEMINI_MODEL` | Optional | Model ID (default: `gemini-2.5-flash`) |
| `TAVILY_API_KEY` | ✅ for destination research | Tavily web search key; obtain one at [Tavily](https://tavily.com/) |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | `true` / `false` |
| `GEOAPIFY_API_KEY` | Future | Location and places |

> ⚠️ **Never commit `.env` to version control.** It is listed in `.gitignore`.

## Destination Research Agent (Phase 4)

The Destination Research Agent in `backend/agents/destination_research_agent.py`
uses focused Tavily searches for attractions and supported admission information,
local cuisine and culture, seasonal guidance, and safety/travel tips. It passes
the search evidence and supplied trip context to the cached Gemini client from
`backend/llm.py`, then validates Gemini's JSON response with the existing
`DestinationResearch` and `Attraction` Pydantic models. Only validated research
is returned in the `destination_research` shared-state field. If a fact or cost
is not supported by the search results, that value should remain unavailable.

The current schema groups local food, safety considerations, and practical tips
within `DestinationResearch.local_culture`; source URLs are retained in
`research_sources`. This agent is implemented as a standalone state-update
function only. No LangGraph workflow, other agents, or external weather/maps
integrations are part of this phase.

Configure `TAVILY_API_KEY` in `.env` using the placeholder in `.env.example` as
a reminder to supply your own value. The mocked agent tests do not call either
external API and need no credentials:

```bash
python -m pytest tests/test_destination_research_agent.py -v
python -m pytest tests/ -v
```

## Itinerary Planning Agent (Phase 5)

The standalone Itinerary Planning Agent in
`backend/agents/itinerary_planning_agent.py` reads validated trip requirements
and the existing `destination_research` field from `TravelState`. It requires a
positive duration (or dates from which one can be derived) and researched
attractions, then uses Gemini to draft a day-by-day schedule. Gemini receives
the available interests, budget, party size, dates, destination research, and
the deterministic geographic recommendations. The result is checked against
the existing `ItineraryDay` schema and must cover every trip day with the
expected dates.

The route utilities in `backend/tools/route_optimizer.py` use Haversine
great-circle distance for straight-line proximity. A standard-library DBSCAN
implementation groups attractions within a configurable radius; isolated
points remain separate and unlocated attractions remain itinerary candidates.
Nearby groups are distributed among travel days, with large groups split when
they exceed the soft daily visit cap. A Nearest Neighbor route seed followed by
2-opt improves the visiting order for attractions with verified coordinates.
No coordinates are inferred from attraction names.

The resulting `ItineraryDay` data includes the attraction sequence, an explicit
geographic optimization status, optional estimated straight-line distance, and
notes for attractions without coordinate data. Straight-line distance is not
road distance and does not predict actual travel time; no road routing or paid
geocoding service is used. Days without usable coordinates are marked as
geographically unavailable, and a plan with partial/missing coordinate data is
reported as limited optimization.

Run the new deterministic tests without API credentials:

```bash
python -m pytest tests/test_route_optimizer.py -v
python -m pytest tests/test_itinerary_planning_agent.py -v
```

## Budget Planning Agent (Phase 6)

The Budget Planning Agent in `backend/agents/budget_planning_agent.py` uses only
Python and `Decimal`; it does not call Gemini, another LLM, Tavily, or a booking
service. It reads trip requirements and explicit `budget_inputs` from
`TravelState`, validates the input with Pydantic, and returns an auditable
`budget_breakdown` partial state update. It does not change the itinerary or
destination research, and it does not book hotels, purchase tickets, or make
payments.

Supported categories are accommodation, food, local transportation, intercity
transportation, activities, and separately named other expenses. Prices carry
their currency, pricing unit, source name/URL where available, price type, and
verification status. Published prices need a source and verified status;
manually supplied estimates must be approved. The agent performs no automatic
currency conversion and rejects mixed currencies rather than adding them.

Calculations preserve decimal precision until final monetary rounding to two
places using half-up rounding. Accommodation requires explicit rooms and nights;
food uses travelers times trip days; local transport uses the stated day, trip,
vehicle, or vehicle-trip unit; intercity transport uses explicit traveler,
trip, or vehicle counts; tickets use applicable travelers times explicit visits.
Other expenses use an explicit flat total, item quantity, or the supported
person-based units. Local and intercity transportation remain distinct line
items and categories.

The emergency buffer defaults visibly to 0%; it is never silently added. When a
category lacks a reliable supplied price, it remains unknown and appears in
`missing_categories`. Known amounts can still be subtotaled, but the result is
marked partial and any comparison against the user budget is provisional. No
price is inferred from general research prose or generated by an AI model.

Run the deterministic tests offline (no model/API credentials are needed):

```bash
python -m pytest tests/test_budget_planning_agent.py -v
python -m pytest tests/ -v -k "not integration"
```

## Travel Assistant Agent (Phase 7)

The Travel Assistant Agent in `backend/agents/travel_assistant_agent.py`
provides structured weather, transportation/logistics, categorized packing
recommendations, and practical tips in `TravelState.travel_information`. It
considers the request, destination research, itinerary, and budget when present;
itinerary and budget are optional inputs.

Weather uses Open-Meteo's geocoding and forecast APIs through
`backend/tools/weather_tool.py`. The service requires no API key or additional
dependency. Returned daily observations include temperature, WMO condition,
precipitation, provider, and retrieval time. Forecast horizon gaps, dates in the
past, lookup failures, and provider errors are marked unavailable; no historical
climate average is presented as an actual forecast.

Transportation, local customs, and practical destination tips may be researched
with the existing Tavily search utility. The agent preserves URLs from both
destination research and its own search results. The configured Gemini model is
used only to organize and personalize packing and practical advice against that
evidence. It cannot provide weather readings, fares, schedules, laws, or new
source URLs; unsupported citations cause deterministic fallback advice instead.
General advice is explicitly labeled, and unavailable external information does
not prevent the other sections from being returned.

Test this agent offline with mocked Open-Meteo, Tavily, and Gemini responses:

```bash
python -m pytest tests/test_travel_assistant_agent.py -v
python -m pytest tests/ -v -k "not integration"
```

## Travel Assistant Agent (Phase 7)

The standalone Travel Assistant Agent in
`backend/agents/travel_assistant_agent.py` adds structured weather, transportation
and logistics, categorized packing recommendations, and practical travel tips
to the `travel_information` field in `TravelState`. It reads the request,
destination research, itinerary, and budget when available. Missing itinerary
or budget information is optional and does not prevent assistance.

Weather uses the Open-Meteo geocoding and forecast APIs through
`backend/tools/weather_tool.py`. The provider requires no API key or new
dependency. It returns daily forecast conditions, temperatures, precipitation,
the provider/retrieval source, and unavailable dates. Forecast dates outside the
provider horizon are explicitly unavailable; historical climate averages are
never substituted. If dates, geocoding, or the provider are unavailable, no
weather values are fabricated and packing falls back to general suggestions.

Tavily search reuses the existing `TAVILY_API_KEY` configuration to research
local transit, intercity logistics, customs, and practical destination tips.
Retrieved URLs and titles are retained. Information is labeled as verified
source-backed, general guidance, forecast-based, or unavailable; unverified
services, fares, schedules, laws, or safety claims are not asserted.

The existing cached Gemini client may organize and personalize packing and
practical recommendations only when forecast or web evidence is available. It
does not generate or edit weather readings, and recommendations citing a web
source must refer to a retrieved URL. Invalid model output falls back to
conservative deterministic guidance while preserving provider weather data.
There is no new weather API key to configure in `.env`.

Run the mocked assistant tests or complete offline suite without API credentials:

```bash
python -m pytest tests/test_travel_assistant_agent.py -v
python -m pytest tests/ -v -k "not integration"
```

---

## Gemini LLM Integration (Phase 2)

### Overview

Phase 2 establishes a reusable connection to **Google Gemini** via LangChain's `ChatGoogleGenerativeAI`. All future specialised agents will share a single, cached LLM instance through the `get_llm()` function in `backend/llm.py`.

**Architecture:**

```
backend/config.py          ← Loads .env, validates keys, exposes Settings
        ↓
backend/llm.py             ← get_llm() → cached ChatGoogleGenerativeAI
        ↓
backend/agents/*.py        ← (Phase 3+) All agents call get_llm()
```

### Obtaining a Google AI Studio API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **"Create API key"**.
4. Copy the key and add it to your `.env` file:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

### Configuring the Model

The Gemini model is configurable via the `GEMINI_MODEL` environment variable. The default is `gemini-2.5-flash`.

```env
# In .env
GEMINI_MODEL=gemini-2.5-flash
```

Supported models include `gemini-2.5-flash`, `gemini-2.5-pro`, and newer variants. See [Google's model documentation](https://ai.google.dev/gemini-api/docs/models) for the latest list.

### Running the Interactive Gemini Test

```bash
python -m scripts.test_gemini
```

This script initialises the model, accepts a prompt from the terminal, and displays the Gemini-generated response.

### Running the Gemini Tests

```bash
# Run all Gemini tests (mocked tests always run; integration skipped without key)
python -m pytest tests/test_gemini.py -v

# Run only mocked (offline) tests
python -m pytest tests/test_gemini.py -v -k "not integration"

# Run only the real integration test (requires GOOGLE_API_KEY)
python -m pytest tests/test_gemini.py -v -k "integration"
```

---

## Running the Application

### Verify Setup (Phase 1)

```bash
python verify_setup.py
```

### Start the Backend (Phase 2+)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Frontend (Phase 3+)

```bash
streamlit run frontend/app.py
```

---

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Project foundation, structure, configuration | ✅ Complete |
| **Phase 2** | Gemini LLM integration and reusable LLM module | ✅ Complete |
| **Phase 3** | Shared state management and Pydantic schemas | ✅ Complete |
| **Phase 4** | Destination Research Agent and Tavily search tool | ✅ Complete |
| **Phase 5** | Itinerary Planning Agent with geographic route optimization | ✅ Complete |
| **Phase 6** | Python-only deterministic Budget Planning Agent | ✅ Complete |
| **Phase 7** | Travel Assistant Agent | ✅ Complete |
| **Phase 8** | Validation Agent + Response Generator | ⏳ Planned |
| **Phase 9** | FastAPI REST endpoints | ⏳ Planned |
| **Phase 10** | Streamlit conversational frontend | ⏳ Planned |
| **Phase 11** | SQLite persistence, session memory | ⏳ Planned |
| **Phase 12** | LangSmith tracing, monitoring, evaluation | ⏳ Planned |
| **Phase 13** | Deployment, CI/CD, production hardening | ⏳ Planned |

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

---

## License

MIT License – see [LICENSE](LICENSE) for details.
