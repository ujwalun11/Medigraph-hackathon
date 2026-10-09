"""
agents/graph_agent.py
──────────────────────
FalkorDB Cypher QA Agent for MediGraph.
Accepts natural language clinical questions and translates them
to Cypher queries executed on FalkorDB.
"""

import os
from falkordb import FalkorDB
import config


class FalkorGraphAgent:
    def __init__(self):
        self.host = getattr(config, "FALKORDB_HOST", "localhost")
        self.port = getattr(config, "FALKORDB_PORT", 6379)
        self.graph_name = getattr(config, "FALKORDB_GRAPH_NAME", "clinical_graph")
        try:
            self.client = FalkorDB(host=self.host, port=self.port)
            self.graph = self.client.select_graph(self.graph_name)
            print(f"[FalkorGraphAgent] Connected to FalkorDB {self.host}:{self.port}/{self.graph_name}")
        except Exception as e:
            print(f"[FalkorGraphAgent] Connection failed: {e}")
            self.graph = None

        self._init_llm()

    def _init_llm(self):
        backend = getattr(config, "LLM_BACKEND", "gemini").lower()
        if backend == "gemini" or (getattr(config, "GOOGLE_API_KEY", "") and config.AGENT_MODEL.lower().startswith("gemini")):
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = config.AGENT_MODEL if config.AGENT_MODEL.lower().startswith("gemini") else "gemini-2.0-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=config.GOOGLE_API_KEY,
                temperature=config.LLM_TEMPERATURE
            )
        else:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                base_url=config.OLLAMA_BASE_URL,
                model=config.AGENT_MODEL,
                temperature=config.LLM_TEMPERATURE
            )

    def query(self, question: str) -> dict:
        """Translates a natural language medical question to FalkorDB Cypher and runs it."""
        if not self.graph:
            return {"result": "FalkorDB connection is offline.", "cypher": "", "records": []}

        prompt = f"""You are a Cypher query generator for a FalkorDB medical knowledge graph.
Schema:
- (:Symptom {{name: STRING}})-[:INDICATES]->(:Condition {{name: STRING}})
- (:Medication {{name: STRING}})-[:TREATS]->(:Condition {{name: STRING}})
- (:Medication {{name: STRING}})-[:CONTRAINDICATED_FOR]->(:Condition {{name: STRING}})

Rules:
1. Generate valid OpenCypher supported by FalkorDB.
2. Case-insensitive string matching: use toLower(s.name) CONTAINS toLower('term').
3. Output ONLY the raw Cypher query, no markdown blocks, no commentary.

Question: {question}
"""
        response = self.llm.invoke(prompt)
        raw_content = response.content
        if isinstance(raw_content, list):
            raw_content = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content])
        cypher = str(raw_content).strip().replace("```cypher", "").replace("```", "").strip()

        try:
            res = self.graph.query(cypher)
            columns = [col[1] if isinstance(col, (list, tuple)) and len(col) > 1 else str(col) for col in res.header]
            records = [dict(zip(columns, row)) for row in res.result_set]
            return {
                "result": f"Found {len(records)} results.",
                "cypher": cypher,
                "records": records
            }
        except Exception as e:
            return {
                "result": f"FalkorDB Query error: {e}",
                "cypher": cypher,
                "records": []
            }
