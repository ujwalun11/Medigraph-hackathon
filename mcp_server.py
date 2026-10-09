"""
mcp_server.py
─────────────
Model Context Protocol (MCP) Server for MediGraph.
Exposes the FalkorDB clinical knowledge tools to any standard MCP client
(e.g., Claude Desktop, Continue, Cursor, or external LLMs) via SSE.
"""

import json
import config
from agents.agentic_triage import (
    get_conditions_for_symptoms,
    get_discriminating_symptoms,
    get_contraindications,
    get_treatments,
    check_patient_medication_safety,
    ask_clarifying_question,
    find_shortest_clinical_path,
    get_central_clinical_hubs,
    get_connected_clinical_community,
)

try:
    from mcp.server.mcpserver import MCPServer
    mcp = MCPServer("MediGraph-FalkorDB")
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("MediGraph-FalkorDB")


@mcp.tool()
def search_conditions(symptoms: list[str]) -> str:
    """Find candidate conditions associated with symptoms using hybrid search on FalkorDB."""
    return get_conditions_for_symptoms.invoke({"symptoms": symptoms})


@mcp.tool()
def find_discriminating_symptoms(condition_a: str, condition_b: str) -> str:
    """Find unique symptoms distinguishing condition_a from condition_b in FalkorDB."""
    return get_discriminating_symptoms.invoke({"condition_a": condition_a, "condition_b": condition_b})


@mcp.tool()
def find_contraindications(condition: str) -> str:
    """Find medications contraindicated for a confirmed condition in FalkorDB."""
    return get_contraindications.invoke({"condition": condition})


@mcp.tool()
def find_treatments(condition: str) -> str:
    """Find recommended standard treatments for a condition in FalkorDB."""
    return get_treatments.invoke({"condition": condition})


@mcp.tool()
def verify_medication_safety(medication: str, patient_conditions: list[str]) -> str:
    """Verify if a medication is contraindicated for any of the patient's conditions."""
    return check_patient_medication_safety.invoke({
        "medication": medication,
        "patient_conditions": patient_conditions
    })


@mcp.tool()
def ask_patient_clarification(question: str) -> str:
    """Ask the patient a targeted clinical clarifying question when ambiguous symptoms exist."""
    return ask_clarifying_question.invoke({"question": question})


@mcp.tool()
def shortest_path_traversal(source_entity: str, target_entity: str) -> str:
    """Graph Algorithm: Compute the shortest multi-hop causal chain between two clinical entities."""
    return find_shortest_clinical_path.invoke({
        "source_entity": source_entity,
        "target_entity": target_entity
    })


@mcp.tool()
def symptom_centrality_ranking(symptoms: list[str]) -> str:
    """Graph Algorithm: Compute degree centrality scores to identify primary acute symptom hubs."""
    return get_central_clinical_hubs.invoke({"symptoms": symptoms})


@mcp.tool()
def comorbidity_community_detection(condition: str) -> str:
    """Graph Algorithm: Detect comorbidity clusters and shared manifestation communities around a condition."""
    return get_connected_clinical_community.invoke({"condition": condition})



if __name__ == "__main__":
    print(f"[MCP Server] Starting MediGraph MCP Server on port {config.MCP_PORT}...")
    mcp.run(transport="sse")
