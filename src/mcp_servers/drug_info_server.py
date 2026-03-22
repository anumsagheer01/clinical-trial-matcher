"""
MCP Server: Drug Information (via openFDA API)

This server gives Claude access to drug information from the FDA.
When checking if a patient is eligible for a trial, Claude often
needs to know about drug interactions and contraindications.

Tools:
1. get_drug_info — Get label information for a drug
2. check_drug_interactions — Check for known interactions between drugs
3. get_drug_warnings — Get warnings and contraindications for a drug

Data source: openFDA API (free, no API key required)
Docs: https://open.fda.gov/apis/drug/
"""

import json
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DrugInformation")

OPENFDA_BASE = "https://api.fda.gov/drug"


@mcp.tool()
def get_drug_info(drug_name: str) -> str:
    """
    Get FDA label information for a specific drug.

    Returns key details: generic name, brand name, purpose,
    active ingredients, dosage forms, and route of administration.

    Args:
        drug_name: Name of the drug (generic or brand name).
                   Examples: "metformin", "lisinopril", "Ozempic"

    Returns:
        JSON string with drug label information.
    """
    try:
        response = requests.get(
            f"{OPENFDA_BASE}/label.json",
            params={
                "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                "limit": 1,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return json.dumps({"error": f"No FDA data found for '{drug_name}'"})

        data = response.json()
        results = data.get("results", [])
        if not results:
            return json.dumps({"error": f"No FDA data found for '{drug_name}'"})

        label = results[0]
        openfda = label.get("openfda", {})

        return json.dumps({
            "generic_name": openfda.get("generic_name", ["Unknown"])[0],
            "brand_name": openfda.get("brand_name", ["Unknown"])[0],
            "purpose": _get_first(label, "purpose"),
            "indications_and_usage": _get_first(label, "indications_and_usage")[:500],
            "active_ingredient": _get_first(label, "active_ingredient"),
            "dosage_forms": openfda.get("dosage_form", []),
            "route": openfda.get("route", []),
            "pharm_class": openfda.get("pharm_class_epc", []),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to fetch drug info: {str(e)}"})


@mcp.tool()
def check_drug_interactions(drug_name: str) -> str:
    """
    Check for known drug interactions for a specific medication.

    This is important when matching patients to trials — if a patient
    is on a medication that interacts with the trial's intervention,
    they may be excluded.

    Args:
        drug_name: Name of the drug to check interactions for.

    Returns:
        JSON string with interaction warnings.
    """
    try:
        response = requests.get(
            f"{OPENFDA_BASE}/label.json",
            params={
                "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                "limit": 1,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return json.dumps({"error": f"No data found for '{drug_name}'"})

        data = response.json()
        results = data.get("results", [])
        if not results:
            return json.dumps({"error": f"No data found for '{drug_name}'"})

        label = results[0]

        return json.dumps({
            "drug": drug_name,
            "drug_interactions": _get_first(label, "drug_interactions")[:1000],
            "contraindications": _get_first(label, "contraindications")[:1000],
            "warnings": _get_first(label, "warnings")[:1000],
            "precautions": _get_first(label, "precautions")[:500],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to check interactions: {str(e)}"})


@mcp.tool()
def get_drug_warnings(drug_name: str) -> str:
    """
    Get warnings, boxed warnings, and adverse reactions for a drug.

    Boxed warnings (sometimes called "black box warnings") are the
    FDA's strongest safety warnings. These are critical for assessing
    whether a patient should participate in a trial.

    Args:
        drug_name: Name of the drug.

    Returns:
        JSON string with safety warnings.
    """
    try:
        response = requests.get(
            f"{OPENFDA_BASE}/label.json",
            params={
                "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                "limit": 1,
            },
            timeout=10,
        )

        if response.status_code != 200:
            return json.dumps({"error": f"No data found for '{drug_name}'"})

        data = response.json()
        results = data.get("results", [])
        if not results:
            return json.dumps({"error": f"No data found for '{drug_name}'"})

        label = results[0]

        return json.dumps({
            "drug": drug_name,
            "boxed_warning": _get_first(label, "boxed_warning")[:500],
            "warnings_and_cautions": _get_first(label, "warnings_and_cautions")[:500],
            "adverse_reactions": _get_first(label, "adverse_reactions")[:500],
            "overdosage": _get_first(label, "overdosage")[:300],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Failed to get warnings: {str(e)}"})


def _get_first(label, field):
    """Helper: FDA labels store most fields as arrays. Get the first element."""
    value = label.get(field, [])
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, str):
        return value
    return ""


if __name__ == "__main__":
    mcp.run()