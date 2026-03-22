"""
MCP Server: Clinical Trial Search

This server exposes my OpenSearch trial database as MCP tools.
When Claude (or any MCP-compatible agent) connects to this server,
it can discover and call these tools:

1. search_trials - Find trials by condition, age, location, etc.
2. get_trial_by_id - Get full details for a specific trial
3. get_trial_eligibility - Get eligibility criteria for a trial
4. count_trials - Count how many trials match certain criteria

Each tool has a clear description so the AI knows when and how to use it.
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from opensearchpy import OpenSearch

from src.search.index_config_v2 import INDEX_NAME
from src.search.hybrid_search import hybrid_search

load_dotenv()

# Create the MCP server instance
# The name shows up when agents discover this server
mcp = FastMCP("ClinicalTrialSearch")


def get_opensearch_client():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True, use_ssl=False,
        verify_certs=False, ssl_show_warn=False,
    )


@mcp.tool()
def search_trials(
    query: str,
    age: int | None = None,
    sex: str | None = None,
    country: str | None = None,
    state: str | None = None,
    phase: str | None = None,
    max_results: int = 10,
) -> str:
    """
    Search for clinical trials matching a patient's condition.

    Use this tool when you need to find clinical trials that a patient
    might be eligible for. You can search by medical condition, and
    optionally filter by the patient's age, sex, and location.

    Args:
        query: The medical condition or description to search for.
               Examples: "type 2 diabetes", "breast cancer",
               "chronic kidney disease with hypertension"
        age: Patient's age in years (filters for trials accepting this age)
        sex: Patient's sex — "Male" or "Female" (filters accordingly)
        country: Country to search in (e.g., "United States")
        state: State/province (e.g., "Maryland", "California")
        phase: Trial phase filter — "PHASE1", "PHASE2", "PHASE3", "PHASE4"
        max_results: Number of results to return (default 10, max 20)

    Returns:
        JSON string with matching trials, each containing:
        nct_id, title, conditions, eligibility info, and locations.
    """
    max_results = min(max_results, 20)  # Cap at 20 to keep responses manageable

    results = hybrid_search(
        query_text=query,
        min_age=age,
        max_age=age,
        sex=sex,
        country=country,
        state=state,
        phase=phase,
        status="RECRUITING",  # Only show trials that are currently recruiting
        page_size=max_results,
    )

    # Format results for Claude — only include the fields it needs
    # to reason about eligibility. Sending everything would waste tokens.
    formatted = []
    for trial in results["trials"]:
        elig = trial.get("eligibility", {})
        locs = trial.get("locations", [])

        # Pick top 3 locations to show
        top_locs = []
        for loc in locs[:3]:
            loc_str = f"{loc.get('facility', '')}, {loc.get('city', '')}, {loc.get('state', '')}"
            top_locs.append(loc_str.strip(", "))

        formatted.append({
            "nct_id": trial["nct_id"],
            "title": trial["brief_title"],
            "conditions": trial["conditions"],
            "phase": trial.get("phases", []),
            "enrollment": trial.get("enrollment_count", 0),
            "age_range": f"{elig.get('min_age', 'any')} - {elig.get('max_age', 'any')} years",
            "sex": elig.get("sex", "All"),
            "summary": trial.get("brief_summary", "")[:300],  # Truncate to save tokens
            "locations": top_locs,
            "relevance_score": round(trial.get("_score", 0), 4),
        })

    output = {
        "total_matches": results["total"],
        "showing": len(formatted),
        "search_time_ms": results["took_ms"],
        "trials": formatted,
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def get_trial_details(nct_id: str) -> str:
    """
    Get complete details for a specific clinical trial by its NCT ID.

    Use this tool when you need full information about a specific trial,
    including detailed eligibility criteria, all locations, and
    intervention details. Useful after finding a trial via search_trials.

    Args:
        nct_id: The NCT identifier (e.g., "NCT04280705")

    Returns:
        JSON string with complete trial details, or error message if not found.
    """
    client = get_opensearch_client()

    try:
        result = client.get(index=INDEX_NAME, id=nct_id)
        trial = result["_source"]

        # Remove the vector field — Claude doesn't need 384 numbers
        trial.pop("search_vector", None)
        trial.pop("search_text", None)

        return json.dumps(trial, indent=2)
    except Exception:
        return json.dumps({"error": f"Trial {nct_id} not found"})


@mcp.tool()
def get_eligibility_criteria(nct_id: str) -> str:
    """
    Get detailed eligibility criteria for a specific clinical trial.

    Use this when you need to determine if a patient qualifies for
    a specific trial. Returns inclusion criteria, exclusion criteria,
    age range, and sex requirements.

    Args:
        nct_id: The NCT identifier (e.g., "NCT04280705")

    Returns:
        JSON string with eligibility details.
    """
    client = get_opensearch_client()

    try:
        result = client.get(index=INDEX_NAME, id=nct_id)
        trial = result["_source"]
        elig = trial.get("eligibility", {})

        return json.dumps({
            "nct_id": nct_id,
            "title": trial.get("brief_title", ""),
            "min_age": elig.get("min_age"),
            "max_age": elig.get("max_age"),
            "sex": elig.get("sex", "All"),
            "inclusion_criteria": elig.get("inclusion_criteria", []),
            "exclusion_criteria": elig.get("exclusion_criteria", []),
            "full_criteria_text": elig.get("criteria_text", ""),
        }, indent=2)
    except Exception:
        return json.dumps({"error": f"Trial {nct_id} not found"})


@mcp.tool()
def count_trials(
    condition: str | None = None,
    country: str | None = None,
    phase: str | None = None,
) -> str:
    """
    Count how many recruiting trials match given criteria.

    Useful for giving patients a sense of how many options exist
    for their condition in their area.

    Args:
        condition: Medical condition to search for
        country: Country to filter by
        phase: Trial phase to filter by
    """
    results = hybrid_search(
        query_text=condition,
        country=country,
        phase=phase,
        status="RECRUITING",
        page_size=1,  # Only need the count, not the actual results
    )

    return json.dumps({
        "condition": condition or "all",
        "country": country or "worldwide",
        "phase": phase or "all phases",
        "count": results["total"],
    })


# This is how MCP servers run, they wait for connections
if __name__ == "__main__":
    mcp.run()