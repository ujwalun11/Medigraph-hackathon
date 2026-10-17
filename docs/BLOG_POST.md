# Building MediGraph: Zero-Hallucination Clinical Triage Using FalkorDB, LangGraph, and MCP

> *How we built an autonomous, explainable clinical decision-support agent powered by in-memory GraphBLAS linear algebra, Model Context Protocol (MCP), and 5-layer HIPAA privacy guardrails.*

[![GitHub](https://img.shields.io/badge/GitHub-Medigraph--hackathon-181717?logo=github)](https://github.com/ujwalun11/Medigraph-hackathon)
[![Video Demo](https://img.shields.io/badge/Demo-Google_Drive-34A853?logo=google-drive)](https://drive.google.com/file/d/1NpWncTtNR7h6FOIIT2Zaq1VD8xYasGdn/view?usp=sharing)

---

## The Silent Danger of Naive Medical LLMs

Imagine a 75-year-old patient who falls down the stairs, sprains his knee, and asks an AI triage assistant:
> *"My knee is swollen and throbbing with severe pain. Can I take Ibuprofen 800mg to reduce swelling? I also feel slight dizziness and chest fluttering."*

A standard Large Language Model (or naive vector RAG system) will typically respond:
> *"Yes, Ibuprofen is an anti-inflammatory that helps with acute joint sprains."*

In clinical medicine, **this advice can be fatal.** 

If that patient has a background history of **Atrial Fibrillation** and takes an oral anticoagulant (like **Apixaban** or **Warfarin**), adding high-dose Ibuprofen causes severe, potentially life-threatening gastrointestinal hemorrhage.

Standard LLMs fail because they evaluate acute complaints in isolation. They lack relational reasoning across complex medical histories, and vector embeddings alone cannot trace multi-hop adverse risk paths.

To solve this, we built **MediGraph** — an autonomous clinical triage agent built on **FalkorDB**, **LangGraph ReAct**, and the **Model Context Protocol (MCP)**.

---

## Why Graph-RAG over Vector Databases?

Medical knowledge is inherently **relational and causal**, not just semantic:
- Symptoms **indicate** conditions (`:INDICATES`)
- Medications **treat** conditions (`:TREATS`)
- Medications are strictly **contraindicated** for specific diseases (`:CONTRAINDICATED_FOR`)

```mermaid
graph LR
    S[Symptom] -->|:INDICATES| C[Condition]
    M[Medication] -->|:TREATS| C
    M -->|:CONTRAINDICATED_FOR| C
```

While vector databases search by textual similarity, **FalkorDB** allows our agent to execute deterministic graph algorithms:
1. **Multi-Hop Causal Chains:** Discovers indirect interactions that vector search misses.
2. **Topological Hub Centrality:** Distinguishes acute life-threatening symptoms from peripheral noise.
3. **Deterministic Safety:** Ensures 100% reproducible drug safety checks backed by clinical ontologies.

---

## Why FalkorDB? Sub-15ms In-Memory Latency for Agent Loops

In our autonomous ReAct (Reason + Act) loop, the agent executes **4 to 7 tool calls per patient encounter**:
1. Prioritize symptoms via centrality
2. Query candidate diseases
3. Perform differential set-difference disambiguation
4. Ask clarifying questions
5. Traverse multi-hop contraindication paths
6. Retrieve guideline treatments

With traditional disk-bound or JVM-heavy graph databases, each hop adds 100–300ms, making multi-step agent loops unacceptably sluggish for real-time triage.

**FalkorDB changes the game.** By representing property graphs as sparse matrices and leveraging **GraphBLAS linear algebra** directly in-memory, complex multi-hop Cypher queries execute in **sub-15 milliseconds**. This allows the agent to reason, backtrack, and query the graph multiple times while keeping the interactive triage experience instant and smooth.

---

## System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │    Patient / Clinician Web Dashboard UI     │
                    └──────────────────────┬───────────────────────┘
                                           │ POST /api/triage/agentic
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        5-LAYER HIPAA PRIVACY GATEKEEPER                                 │
│   Layer 1: Pre-LLM Masking  ──►  Layer 2: Salted SHA-256 Tokenization                  │
│   Layer 3: Output Scanner   ──►  Layer 4: Identity Vault  ──► Layer 5: Audit Log       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Sanitized Symptoms & History
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH AUTONOMOUS ReAct AGENT                                │
│                                                                                        │
│   ┌──────────────┐     Decide Tool     ┌──────────────┐   Clarification   ┌────────┐  │
│   │  Agent Node  │ ──────────────────► │  Tool Node   │ ────────────────► │ Paused │  │
│   │ (Gemini 2.5) │ ◄────────────────── │  (FalkorDB)  │                   │ User Q │  │
│   └──────┬───────┘     Observation     └──────────────┘                   └────┬───┘  │
│          │                                                                     │      │
│          │ Complete                                      User Clarification Answer    │
│          ▼                                                                     │      │
│   ┌──────────────┐                                                             │      │
│   │ Final Report │ ◄───────────────────────────────────────────────────────────┘      │
│   └──────────────┘                                                                    │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Tool Calling Handshake via MCP (SSE)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      MCP SERVER (Port 8010 over SSE Transport)                         │
│   • search_conditions              • find_discriminating_symptoms                     │
│   • find_contraindications         • find_treatments                                  │
│   • verify_medication_safety       • ask_patient_clarification                        │
│   • shortest_path_traversal        • symptom_centrality_ranking                        │
│   • comorbidity_community_detection                                                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ OpenCypher (GraphBLAS Linear Algebra)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FALKORDB IN-MEMORY KNOWLEDGE GRAPH                              │
│   (Symptom) ────[:INDICATES]────► (Condition) ◄────[:CONTRAINDICATED_FOR]──── (Med)    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Graph Algorithms as Agent Tools

Rather than expecting the LLM to write complex Cypher queries from scratch (which frequently syntax-error), we packaged advanced graph traversals into modular tools exposed via the **Model Context Protocol (MCP)**:

### 1. Multi-Hop Shortest Path (*"Show the Path Behind the Answer"*)
When explaining why a medication is dangerous, the agent calls `find_shortest_clinical_path`:
```cypher
MATCH (a), (b)
WHERE toLower(a.name) CONTAINS toLower($source) 
  AND toLower(b.name) CONTAINS toLower($target) 
  AND a <> b
MATCH p = (a)-[*1..4]-(b)
RETURN [n IN nodes(p) | n.name] AS path_nodes, 
       [r IN relationships(p) | type(r)] AS rels, 
       length(p) AS hop_count
ORDER BY hop_count ASC LIMIT 1;
```
**Output Proof:**
$$(Ibuprofen) \xrightarrow{CONTRAINDICATED\_FOR} (Atrial\ Fibrillation) \xleftarrow{INDICATES} (Chest\ Pain)$$

### 2. Degree Centrality Hub Analysis
When patients report diffuse complaints (*"fatigued, headache, chest tightness, nausea"*), the agent invokes `get_central_clinical_hubs`. It analyzes out-degree topology in FalkorDB to determine which manifestations are primary acute hubs versus secondary symptoms.

### 3. Comorbidity Community Detection
Diseases rarely occur in isolation. Calling `get_connected_clinical_community` projects a 2-hop bipartite subgraph around candidate conditions to detect shared manifestation clusters (e.g., *CAD $\leftrightarrow$ Myocardial Infarction $\leftrightarrow$ Hypertension*).

### 4. Interactive Disambiguation Sentinel
If the patient's presentation is non-specific between two serious conditions (e.g., *Asthma* vs. *Coronary Artery Disease*), the agent computes the graph set-difference between their symptoms and triggers a sentinel tool `ask_clarifying_question`. The LangGraph state machine **suspends execution**, presents a targeted question to the patient, and resumes state once the patient replies.

---

## 5-Layer HIPAA Privacy Architecture

Clinical AI cannot send raw patient data to external LLMs. We engineered an air-gapped 5-layer privacy gatekeeper:
1. **Pre-LLM Masking:** Presidio and regex tokenize raw names, MRNs, phone numbers, and SSNs into `<NAME_1>`, `<SSN_1>`.
2. **Salted SHA-256 Tokenization:** Irreversibly hashes patient chart identifiers.
3. **Output Leak Scanner:** Audits the generated triage report before rendering to verify zero raw PII is echoed back.
4. **Local Identity Vault:** Stores pseudo-to-real mappings in an isolated, air-gapped store accessible only by authenticated clinicians.
5. **Zero-Retention Audit Trail:** Ephemeral cryptographic logging.

---

## Clinical Safety Benchmarks & Results

We evaluated MediGraph using the **RAGAS (Retrieval Augmented Generation Assessment)** framework across gold-standard clinical cases, alongside an offline 35-patient benchmark dataset:

### RAGAS Evaluation Results
| Metric | Score | Clinical Meaning |
|:---|:---:|:---|
| **Clinical Safety Recall** | **100.0%** | **Zero missed lethal drug contraindications** (100% detection rate for high-risk NSAID/Beta-blocker risks). |
| **RAGAS Faithfulness** | **100.0%** | **Zero hallucination rate**; all diagnostic claims are strictly verifiable against FalkorDB facts. |
| **Triage Urgency Accuracy** | **100.0%** | Prioritization (*Emergency*, *Urgent*, *Routine*) aligned with emergency triage guidelines. |
| **RAGAS Composite Score** | **100.0%** | Comprehensive multi-turn agent evaluation score. |

### 35-Patient Dataset Benchmark (Table 6.3)
| Component | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| **Contraindication Detection** | 80.00% | **100.00%** | **88.89%** |
| **Treatment Recommendation** | 70.15% | **97.92%** | **81.74%** |
| **Clinical Grounding (SapBERT)** | 62.79% | **96.43%** | **76.06%** |

---

## Key Learnings & Takeaways

1. **Graphs are Non-Negotiable for Healthcare AI:** Probabilistic LLMs cannot be trusted with medication safety. Grounding an agent in a deterministic property graph eliminates hallucinations and guarantees explainability.
2. **Speed is Crucial for ReAct Agents:** Agentic loops multiply database query frequency. FalkorDB’s sub-15ms GraphBLAS traversals made complex multi-turn reasoning practical in a real-time web application.
3. **MCP is the Universal Tool Standard:** Exposing graph algorithms via the Model Context Protocol allowed seamless tool sharing between our LangGraph agent, the web UI, and external clients like Claude Desktop and Cursor.
4. **Human-in-the-Loop Disambiguation Prevents Misdiagnosis:** Forcing the agent to pause and ask clarifying questions on ambiguous presentations mirrors real-world emergency triage and dramatically boosts diagnostic accuracy.

---

## Try It Out

- 💻 **GitHub Repository:** [ujwalun11/Medigraph-hackathon](https://github.com/ujwalun11/Medigraph-hackathon)
- 🎬 **Watch Demo Video:** [Google Drive Walkthrough](https://drive.google.com/file/d/1NpWncTtNR7h6FOIIT2Zaq1VD8xYasGdn/view?usp=sharing)
- 🚀 **Quickstart:**
  ```bash
  docker run -d -p 6379:6379 falkordb/falkordb:latest
  pip install -r requirements.txt
  python seed_falkordb.py
  python mcp_server.py &
  python -m uvicorn api:app --port 8008 --reload
  ```
