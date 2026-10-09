"""
config.py - Centralized configuration for MediGraph (Hackathon Submission).
All environment-sensitive values can be overridden via a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# FalkorDB - Graph database engine based on GraphBLAS
# Start: docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH_NAME = os.getenv("FALKORDB_GRAPH_NAME", "clinical_graph")

# LLM Provider
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini" if os.getenv("GOOGLE_API_KEY") else "ollama").lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-3.8-flash")
LLM_TEMPERATURE = 0.0

# Ollama fallback
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-coder:7b")

# LangGraph
LANGGRAPH_RECURSION_LIMIT = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "25"))

# MCP Server
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8010"))

# Triage API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8091"))
