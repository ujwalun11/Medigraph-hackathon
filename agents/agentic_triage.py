"""
agents/agentic_triage.py
────────────────────────
Agentic Clinical Triage using LangGraph ReAct loop + FalkorDB.

Architecture:
  LangGraph StateGraph with FalkorDB tool-calling nodes.
  The LLM (Gemini 2.0 Flash / Qwen) autonomously queries the FalkorDB
  knowledge graph to identify conditions, check contraindications,
  and ask clarifying questions if ambiguous.
"""

import json
import os
import uuid
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import config

# ── FalkorDB Driver & Embeddings (shared, lazy) ──────────────────────────────

_GRAPH = None
_ENCODER = None
_DB_SYMPTOMS = None
_SYMPTOM_EMBEDDINGS = None


def _get_graph():
    """Lazily create a FalkorDB graph client."""
    global _GRAPH
    if _GRAPH is None:
        try:
            from falkordb import FalkorDB
            client = FalkorDB(
                host=getattr(config, "FALKORDB_HOST", "localhost"),
                port=getattr(config, "FALKORDB_PORT", 6379),
            )
            _GRAPH = client.select_graph(getattr(config, "FALKORDB_GRAPH_NAME", "clinical_graph"))
            print("[AgenticTriage] Connected to FalkorDB.")
        except Exception as e:
            print(f"[AgenticTriage] FalkorDB connection failed: {e}")
            _GRAPH = None
    return _GRAPH


def _get_encoder():
    """Lazily load the clinical SapBERT encoder only if explicitly enabled."""
    global _ENCODER
    if _ENCODER is None:
        if os.getenv("USE_SAPBERT", "false").lower() != "true":
            _ENCODER = False
            return None
        try:
            from sentence_transformers import SentenceTransformer
            print("[AgenticTriage] Loading clinical SapBERT encoder...")
            _ENCODER = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        except Exception as e:
            print(f"[AgenticTriage] SapBERT encoder unavailable ({e}), using sparse graph search.")
            _ENCODER = False
    return _ENCODER if _ENCODER is not False else None


def _cache_symptoms(graph, encoder):
    """Cache all symptoms from FalkorDB and precompute embeddings if encoder available."""
    global _DB_SYMPTOMS, _SYMPTOM_EMBEDDINGS
    if _DB_SYMPTOMS is not None:
        return

    _DB_SYMPTOMS = []
    if graph:
        try:
            res = graph.query("MATCH (s:Symptom) RETURN s.name AS name")
            for row in res.result_set:
                _DB_SYMPTOMS.append(row[0])
        except Exception as e:
            print(f"[AgenticTriage] Error fetching symptoms: {e}")

    if _DB_SYMPTOMS and encoder:
        try:
            print(f"[AgenticTriage] Caching {len(_DB_SYMPTOMS)} reference symptoms & generating embeddings...")
            _SYMPTOM_EMBEDDINGS = encoder.encode(_DB_SYMPTOMS)
        except Exception as e:
            print(f"[AgenticTriage] Symptom embedding cache failed ({e}), relying on sparse search.")
            _SYMPTOM_EMBEDDINGS = None


# ── FalkorDB Graph Query Tools ───────────────────────────────────────────────

@tool
def get_conditions_for_symptoms(symptoms: List[str]) -> str:
    """
    Query the FalkorDB clinical knowledge graph to find candidate conditions
    associated with a list of symptoms using Hybrid (Sparse + Dense) Search.
    Returns ranked condition names with matching scores.
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    encoder = _get_encoder()
    _cache_symptoms(graph, encoder)

    condition_counts: Dict[str, float] = {}

    for symptom in symptoms:
        # 1. Sparse / Substring match in FalkorDB
        try:
            query = (
                "MATCH (s:Symptom)-[:INDICATES]->(c:Condition) "
                f"WHERE toLower(s.name) CONTAINS toLower('{symptom}') "
                "RETURN c.name AS condition"
            )
            res = graph.query(query)
            for row in res.result_set:
                cond = row[0]
                condition_counts[cond] = condition_counts.get(cond, 0.0) + 1.0
        except Exception as e:
            print(f"[AgenticTriage] Sparse query error: {e}")

        # 2. Dense / Semantic match using SapBERT
        if _DB_SYMPTOMS and _SYMPTOM_EMBEDDINGS is not None and encoder:
            import numpy as np
            sym_emb = encoder.encode(symptom)
            similarities = np.dot(_SYMPTOM_EMBEDDINGS, sym_emb) / (
                np.linalg.norm(_SYMPTOM_EMBEDDINGS, axis=1) * np.linalg.norm(sym_emb)
            )
            matched_indices = np.where(similarities >= 0.50)[0]
            for idx in matched_indices:
                matched_sym = _DB_SYMPTOMS[idx]
                sim_score = float(similarities[idx])
                try:
                    res_dense = graph.query(
                        f"MATCH (s:Symptom {{name: '{matched_sym}'}})-[:INDICATES]->(c:Condition) "
                        "RETURN c.name AS condition"
                    )
                    for r in res_dense.result_set:
                        cond = r[0]
                        condition_counts[cond] = condition_counts.get(cond, 0.0) + sim_score
                except Exception:
                    pass

    if not condition_counts:
        return "No matching conditions found in graph for those symptoms."

    ranked = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)
    return json.dumps([{"condition": c, "match_score": round(score, 3)} for c, score in ranked])


@tool
def get_discriminating_symptoms(condition_a: str, condition_b: str) -> str:
    """
    Query FalkorDB to find symptoms unique to condition_a vs condition_b.
    Use this to formulate targeted clarifying questions.
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    def get_symptoms(cond: str) -> set:
        res = graph.query(
            f"MATCH (s:Symptom)-[:INDICATES]->(c:Condition {{name: '{cond}'}}) RETURN s.name"
        )
        return {r[0] for r in res.result_set}

    syms_a = get_symptoms(condition_a)
    syms_b = get_symptoms(condition_b)

    return json.dumps({
        f"unique_to_{condition_a}": list(syms_a - syms_b)[:5],
        f"unique_to_{condition_b}": list(syms_b - syms_a)[:5],
    })


@tool
def get_contraindications(condition: str) -> str:
    """
    Query FalkorDB for medications CONTRAINDICATED for a given condition.
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    query = (
        "MATCH (m:Medication)-[:CONTRAINDICATED_FOR]->(c:Condition) "
        f"WHERE toLower(c.name) CONTAINS toLower('{condition}') "
        "RETURN m.name AS medication, c.name AS condition"
    )
    res = graph.query(query)
    records = [{"medication": r[0], "condition": r[1]} for r in res.result_set]

    if not records:
        return f"No contraindications found in graph for '{condition}'."
    return json.dumps(records)


@tool
def get_treatments(condition: str) -> str:
    """
    Query FalkorDB for standard recommended treatments (medications).
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    query = (
        "MATCH (m:Medication)-[:TREATS]->(c:Condition) "
        f"WHERE toLower(c.name) CONTAINS toLower('{condition}') "
        "RETURN m.name AS medication"
    )
    res = graph.query(query)
    meds = [r[0] for r in res.result_set]

    if not meds:
        return f"No standard treatments found in graph for '{condition}'."
    return json.dumps({"condition": condition, "recommended_treatments": meds})


@tool
def check_patient_medication_safety(medication: str, patient_conditions: List[str]) -> str:
    """
    Check whether a medication is contraindicated for ANY of the patient's conditions.
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    risks = []
    for cond in patient_conditions:
        query = (
            "MATCH (m:Medication)-[:CONTRAINDICATED_FOR]->(c:Condition) "
            f"WHERE toLower(m.name) CONTAINS toLower('{medication}') "
            f"  AND toLower(c.name) CONTAINS toLower('{cond}') "
            "RETURN m.name AS med, c.name AS cond"
        )
        res = graph.query(query)
        for r in res.result_set:
            risks.append({"medication": r[0], "contraindicated_for": r[1]})

    if not risks:
        return f"No contraindication found for '{medication}' with given conditions."
    return json.dumps({"warning": "CONTRAINDICATION DETECTED", "details": risks})


@tool
def ask_clarifying_question(question: str) -> str:
    """
    Ask the patient a single targeted clarifying question when ambiguous.
    """
    return json.dumps({"__clarification_needed__": True, "question": question})


@tool
def find_shortest_clinical_path(source_entity: str, target_entity: str) -> str:
    """
    Graph Algorithm (Shortest Path): Traverses the knowledge graph to discover the shortest
    multi-hop causal chain connecting two clinical entities (e.g. a symptom to a drug, or a drug to an adverse condition).
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    query = f"""
    MATCH (a), (b)
    WHERE toLower(a.name) CONTAINS toLower('{source_entity}') AND toLower(b.name) CONTAINS toLower('{target_entity}') AND a <> b
    MATCH p = (a)-[*1..4]-(b)
    RETURN [n IN nodes(p) | n.name] AS path_nodes, [r IN relationships(p) | type(r)] AS rels, length(p) AS hop_count
    ORDER BY hop_count ASC
    LIMIT 1
    """
    try:
        res = graph.query(query)
        if not res.result_set:
            return f"No path found between '{source_entity}' and '{target_entity}' within 4 hops."
        row = res.result_set[0]
        nodes, rels, hops = row[0], row[1], row[2]
        formatted_steps = []
        for i in range(len(rels)):
            formatted_steps.append(f"({nodes[i]}) -[{rels[i]}]-> ({nodes[i+1]})")
        return json.dumps({
            "source": source_entity,
            "target": target_entity,
            "hop_count": hops,
            "path_nodes": nodes,
            "relationships": rels,
            "causal_chain": " -> ".join(formatted_steps)
        })
    except Exception as e:
        return f"Error computing shortest path: {e}"


@tool
def get_central_clinical_hubs(symptoms: List[str]) -> str:
    """
    Graph Algorithm (Degree Centrality): Computes the clinical centrality of reported symptoms
    to prioritize primary acute 'hub' symptoms over peripheral non-specific complaints.
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    # Substring / fuzzy conditions to match DB nodes like 'Chest Pain' for 'chest discomfort'
    match_clauses = []
    for s in symptoms:
        clean = s.lower().replace("discomfort", "pain").strip()
        match_clauses.append(f"toLower(s.name) CONTAINS '{clean}' OR '{clean}' CONTAINS toLower(s.name)")
    
    where_expr = " OR ".join(match_clauses) if match_clauses else "1=1"
    query = f"""
    MATCH (s:Symptom)-[:INDICATES]->(c:Condition)
    WHERE {where_expr}
    RETURN s.name AS symptom, count(c) AS degree_centrality, collect(c.name)[0..3] AS linked_conditions
    ORDER BY degree_centrality DESC
    """
    try:
        res = graph.query(query)
        records = [
            {"symptom": r[0], "centrality_score": r[1], "sample_conditions": r[2]}
            for r in res.result_set
        ]
        if not records:
            return f"No centrality metrics found for symptoms: {symptoms}"
        return json.dumps({"centrality_rankings": records})
    except Exception as e:
        return f"Error computing symptom centrality: {e}"


@tool
def get_connected_clinical_community(condition: str) -> str:
    """
    Graph Algorithm (Community Detection & Comorbidity Subgraphs): Identifies connected disease
    and symptom communities clustered around a given condition to catch multi-condition syndromes.
    """
    graph = _get_graph()
    if not graph:
        return "Graph unavailable."

    query = f"""
    MATCH (c:Condition)<-[:INDICATES]-(s:Symptom)-[:INDICATES]->(related:Condition)
    WHERE toLower(c.name) CONTAINS toLower('{condition}') AND related <> c
    RETURN related.name AS comorbidity_cluster, count(s) AS shared_symptoms_count, collect(s.name)[0..4] AS shared_symptoms
    ORDER BY shared_symptoms_count DESC
    LIMIT 4
    """
    try:
        res = graph.query(query)
        clusters = [
            {"condition": r[0], "shared_manifestations_count": r[1], "shared_symptoms": r[2]}
            for r in res.result_set
        ]
        if not clusters:
            return f"No comorbidity community clusters found for '{condition}'."
        return json.dumps({"primary_condition": condition, "connected_communities": clusters})
    except Exception as e:
        return f"Error detecting comorbidity community: {e}"


# ── LangGraph Workflow ────────────────────────────────────────────────────────

TOOLS = [
    get_conditions_for_symptoms,
    get_discriminating_symptoms,
    get_contraindications,
    get_treatments,
    check_patient_medication_safety,
    ask_clarifying_question,
    find_shortest_clinical_path,
    get_central_clinical_hubs,
    get_connected_clinical_community,
]

TOOL_MAP = {t.name: t for t in TOOLS}


class TriageState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_history: str
    clarification_asked: bool
    clarification_answer: str
    final_report: Optional[Dict]


SYSTEM_PROMPT = """You are an expert clinical triage physician powered by a FalkorDB medical knowledge graph.
Your goal is to analyze the patient's symptoms and generate a structured triage report strictly grounded in graph intelligence tools.

REASONING STRATEGY:
1. Symptom Prioritization: If the patient presents multiple symptoms, call `get_central_clinical_hubs` to rank primary acute hub symptoms, then call `get_conditions_for_symptoms`.
2. Diagnostic Disambiguation: If 2+ candidate conditions are plausible or overlapping (e.g., Asthma vs CAD vs COPD vs Arrhythmia), call `get_discriminating_symptoms` and then call `ask_clarifying_question` with a targeted discriminating question to resolve ambiguity.
3. Multi-Hop Safety & Traversal: For any proposed or patient-mentioned medications, call `find_shortest_clinical_path` and `check_patient_medication_safety` to trace causal chains and verify contraindications against the patient's medical history.
4. Comorbidity & Treatments: Call `get_connected_clinical_community` to check associated disease clusters, then call `get_contraindications` and `get_treatments` for the confirmed condition(s).
5. Final Report: Generate your final report as strictly valid JSON.

IMPORTANT CLINICAL RULES:
- Call `ask_clarifying_question` AT MOST ONCE per triage encounter when symptoms are non-specific or ambiguous.
- Always ground all diagnoses, contraindications, and treatments strictly in tool execution responses.
- "confirmed_conditions": Include ONLY the primary 1 or 2 most likely clinical diagnoses matching the current acute presentation. Do NOT dump the entire list of differential candidates or all historical conditions here.
- "contraindications_found": Collect and report ALL detected medication contraindications from tool outputs.
- "recommended_treatments": Include safe, guideline-directed treatments retrieved from the graph.

FINAL REPORT FORMAT (output strictly valid JSON as your final answer):
{
  "urgency_level": "Emergency | Urgent | Routine",
  "urgency_score": 1-10,
  "suggested_specialty": "string",
  "reasoning_steps": ["step1", "step2"],
  "summary": {
    "symptoms": ["string"],
    "duration": "string",
    "red_flags": ["string"],
    "clinical_notes": "string"
  },
  "graph_grounded_facts": {
    "confirmed_conditions": ["string"],
    "contraindications_found": ["string"],
    "recommended_treatments": ["string"]
  }
}
"""


FALLBACK_MODELS = [
    getattr(config, "AGENT_MODEL", "gemini-3.6-flash"),
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


def build_llm(model_override: Optional[str] = None):
    backend = getattr(config, "LLM_BACKEND", "gemini").lower()
    selected_model = model_override or getattr(config, "AGENT_MODEL", "gemini-3.6-flash")
    if backend == "gemini" or (getattr(config, "GOOGLE_API_KEY", "") and selected_model.lower().startswith("gemini")):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=selected_model,
            api_key=config.GOOGLE_API_KEY,
            temperature=config.LLM_TEMPERATURE,
        ).bind_tools(TOOLS)
    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.AGENT_MODEL,
        temperature=config.LLM_TEMPERATURE,
    ).bind_tools(TOOLS)


def agent_node(state: TriageState) -> TriageState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    if state.get("clarification_asked") and state.get("clarification_answer"):
        messages.append(HumanMessage(content=f"Patient answered: {state['clarification_answer']}"))

    # Try models in sequence with graceful fallback if 429 quota or rate-limit hit
    last_err = None
    seen_models = set()
    for model_name in FALLBACK_MODELS:
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        try:
            llm = build_llm(model_name)
            response = llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
                print(f"[AgenticTriage] Model {model_name} quota hit, falling back to next available model...")
                last_err = e
                continue
            raise e

    raise last_err or RuntimeError("All Gemini models exhausted.")


def tool_node(state: TriageState) -> TriageState:
    last_message = state["messages"][-1]
    tool_messages = []
    clarification_asked = state.get("clarification_asked", False)

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_fn = TOOL_MAP.get(tool_name)
        result = tool_fn.invoke(tool_args) if tool_fn else f"Unknown tool: {tool_name}"
        if tool_name == "ask_clarifying_question":
            clarification_asked = True
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    return {"messages": tool_messages, "clarification_asked": clarification_asked}


def parse_final_report(state: TriageState) -> TriageState:
    import re
    messages = state.get("messages", [])
    content = messages[-1].content if messages else ""
    if isinstance(content, list):
        content = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return {"final_report": json.loads(match.group(0))}
        except Exception:
            pass

    return {
        "final_report": {
            "urgency_level": "Urgent",
            "urgency_score": 6,
            "suggested_specialty": "Internal Medicine",
            "summary": {"clinical_notes": content[:300]},
            "graph_grounded_facts": {"confirmed_conditions": [], "contraindications_found": [], "recommended_treatments": []}
        }
    }


def route_after_agent(state: TriageState) -> str:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return "parse_report"
    for tc in last.tool_calls:
        if tc["name"] == "ask_clarifying_question":
            return "clarification"
    return "tools"


def clarification_node(state: TriageState) -> TriageState:
    last = state["messages"][-1]
    tool_messages = []
    for tc in last.tool_calls:
        fn = TOOL_MAP.get(tc["name"])
        res = fn.invoke(tc["args"]) if fn else ""
        tool_messages.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
    return {"messages": tool_messages, "clarification_asked": True}


def route_after_clarification(state: TriageState) -> str:
    return "agent" if state.get("clarification_answer") else END


def build_graph():
    g = StateGraph(TriageState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("clarification", clarification_node)
    g.add_node("parse_report", parse_final_report)

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "clarification": "clarification", "parse_report": "parse_report"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("clarification", route_after_clarification, {"agent": "agent", END: END})
    g.add_edge("parse_report", END)
    return g.compile()


GRAPH = build_graph()
_SESSIONS: Dict[str, TriageState] = {}


def run_agentic_triage(symptoms: str, patient_history: str = "", session_id: Optional[str] = None, clarification_answer: Optional[str] = None) -> Dict[str, Any]:
    sid = session_id or str(uuid.uuid4())
    if clarification_answer and sid in _SESSIONS:
        state = _SESSIONS.pop(sid)
        state["clarification_answer"] = clarification_answer
    else:
        init_text = f"Patient symptoms: {symptoms}"
        if patient_history:
            init_text += f"\n\nMedical history: {patient_history}"
        state = {
            "messages": [HumanMessage(content=init_text)],
            "patient_history": patient_history,
            "clarification_asked": False,
            "clarification_answer": "",
            "final_report": None
        }

    res_state = GRAPH.invoke(state, config={"recursion_limit": config.LANGGRAPH_RECURSION_LIMIT})

    # Extract execution steps and tool calls from graph trajectory
    steps = []
    messages = res_state.get("messages", [])
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                steps.append({
                    "type": "tool_call",
                    "tool": tc.get("name", "tool"),
                    "input": tc.get("args", {})
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "type": "observation",
                "content": str(msg.content)
            })

    final_report = res_state.get("final_report", {})

    if res_state.get("clarification_asked") and not res_state.get("clarification_answer"):
        _SESSIONS[sid] = res_state
        extracted_question = "Could you provide more details about your symptoms?"
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                    if isinstance(payload, dict) and payload.get("__clarification_needed__"):
                        extracted_question = payload.get("question", extracted_question)
                        break
                except Exception:
                    pass
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc.get("name") == "ask_clarifying_question":
                        extracted_question = tc.get("args", {}).get("question", extracted_question)
                        break

        return {
            "status": "clarification",
            "session_id": sid,
            "question": extracted_question,
            "steps": steps
        }

    return {
        "status": "complete",
        "session_id": sid,
        "result": final_report,
        "steps": steps
    }
