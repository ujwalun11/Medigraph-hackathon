# 🏥 MediGraph — AI-Powered Clinical Triage with FalkorDB Knowledge Graph

> **Hackathon Submission** — Agentic Clinical Decision Support powered by FalkorDB, LangGraph, and MCP

---

## 🚀 What is MediGraph?

MediGraph is an **agentic clinical triage system** that uses a **FalkorDB property graph** as its knowledge backbone. A LangGraph ReAct agent (Gemini / Qwen) autonomously traverses the graph using specialized medical tools exposed via the **Model Context Protocol (MCP)**.

### Key Features
- 🔬 **Symptom → Condition** mapping with hybrid (sparse + dense SapBERT) search
- 💊 **Contraindication detection** grounded in graph relationships  
- 🩺 **Personalized treatment recommendations**
- 🤔 **Dynamic clarifying questions** when diagnosis is ambiguous
- 🌐 **MCP Server** — tools exposed as a standard MCP SSE endpoint

---

## 🏗️ Architecture

```
Patient Symptoms (text)
        │
        ▼
┌───────────────────┐
│  FastAPI API Layer │  ← POST /triage
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│         LangGraph ReAct Agent              │
│  agent → route → tool_node → loop → report │
└──────────┬─────────────────────────────────┘
           │ calls tools via MCP
           ▼
┌─────────────────────────────────────────────────┐
│           MCP Tool Server  (port 8010)           │
│  - get_conditions_for_symptoms                   │
│  - get_discriminating_symptoms                   │
│  - get_contraindications / get_treatments        │
│  - check_patient_medication_safety               │
└──────────┬──────────────────────────────────────┘
           │ Cypher queries
           ▼
┌──────────────────────┐
│  FalkorDB            │  ← 110 conditions, symptoms, medications
│  Property Graph      │
└──────────────────────┘
```

---

## ⚡ Quick Start

### 1. Start FalkorDB (Docker)
```bash
docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Set GOOGLE_API_KEY (for Gemini) or leave for Ollama
```

### 4. Seed the knowledge graph
```bash
python seed_falkordb.py
```

### 5. Start MCP Tool Server
```bash
python mcp_server.py   # runs on port 8010
```

### 6. Start Triage API
```bash
uvicorn api:app --port 8008 --reload
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/triage` | Run agentic triage for patient symptoms |
| `POST` | `/triage/continue` | Resume session with clarification answer |
| `GET`  | `/graph/conditions` | List all conditions in FalkorDB |
| `GET`  | `/graph/stats` | Graph statistics |
| `GET`  | `/health` | Health check |

---

## 🔌 MCP Integration

Connect any MCP client (Claude Desktop, Continue IDE) to:
```
http://localhost:8010/sse
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Graph DB | **FalkorDB** (Redis-compatible property graph) |
| Agent | **LangGraph** ReAct loop |
| LLM | **Gemini 2.0 Flash** / Qwen3-Coder (Ollama) |
| Embeddings | **SapBERT** (biomedical NLP) |
| Protocol | **MCP** (Model Context Protocol) |
| API | **FastAPI** |
