"""
FastAPI backend for Clinical Trial Matcher.

This is the main API that users (and later, the frontend) will talk to.
It receives a patient description in plain English, uses Claude to
orchestrate the MCP tools, and returns matched trials with explanations.

Endpoints:
- POST /match       — Main endpoint: patient text → matched trials
- GET  /search      — Direct search (no AI, just OpenSearch)
- GET  /trial/{id}  — Get details for a specific trial
- GET  /health      — Health check
"""

import os
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

from src.search.hybrid_search import hybrid_search
from src.mcp_servers.trial_search_server import search_trials, get_trial_details, get_eligibility_criteria
from src.mcp_servers.drug_info_server import get_drug_info, check_drug_interactions

load_dotenv()

app = FastAPI(
    title="Clinical Trial Matcher API",
    description="Match patients to eligible clinical trials using AI",
    version="0.2.0",
)

# Allow frontend to call this API from a different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to my frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# === REQUEST / RESPONSE MODELS ===
# Pydantic models define what the API expects and returns.
# FastAPI uses these for automatic validation and documentation.

class PatientInput(BaseModel):
    """What the user sends to /match"""
    description: str  # Free text like "58 year old male with type 2 diabetes..."
    age: int | None = None
    sex: str | None = None
    country: str | None = None
    state: str | None = None
    medications: list[str] = []  # Current medications the patient is taking
    max_results: int = 5

class SearchInput(BaseModel):
    """What the user sends to /search"""
    query: str
    age: int | None = None
    sex: str | None = None
    country: str | None = None
    state: str | None = None
    phase: str | None = None
    max_results: int = 10


# === CLAUDE AGENT LOGIC ===

def run_agent(patient: PatientInput) -> dict:
    """
    Use Claude to match a patient to clinical trials.

    This is where the magic happens. I give Claude:
    - The patient's description
    - Access to search tools (via function calling)
    - Instructions to search, check eligibility, and explain matches

    Claude decides which tools to call and in what order.
    This is the "agentic" part — Claude is making decisions, not just
    answering a question.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Define the tools Claude can use
    # These match the MCP tool signatures exactly
    tools = [
        {
            "name": "search_trials",
            "description": "Search for clinical trials matching a patient's condition. Returns trials with title, conditions, eligibility, and locations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical condition or description"},
                    "age": {"type": "integer", "description": "Patient age"},
                    "sex": {"type": "string", "enum": ["Male", "Female"]},
                    "country": {"type": "string"},
                    "state": {"type": "string"},
                    "phase": {"type": "string", "enum": ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_trial_details",
            "description": "Get complete details for a specific trial by NCT ID. Use after finding a trial via search.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "NCT identifier like NCT04280705"},
                },
                "required": ["nct_id"],
            },
        },
        {
            "name": "get_eligibility_criteria",
            "description": "Get detailed inclusion/exclusion criteria for a trial. Use to check if patient qualifies.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string"},
                },
                "required": ["nct_id"],
            },
        },
        {
            "name": "get_drug_info",
            "description": "Get FDA label info for a drug. Use to understand medications the patient is taking.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Drug name (generic or brand)"},
                },
                "required": ["drug_name"],
            },
        },
        {
            "name": "check_drug_interactions",
            "description": "Check drug interactions and contraindications. Use to verify patient's medications don't conflict with trial interventions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                },
                "required": ["drug_name"],
            },
        },
    ]

    # The system prompt tells Claude exactly what its job is
    system_prompt = """You are a clinical trial matching assistant. Your job is to help patients find clinical trials they may be eligible for.

Given a patient description, you must:
1. Search for relevant clinical trials using the search_trials tool
2. For the top matches, check detailed eligibility criteria using get_eligibility_criteria
3. If the patient is on medications, check for drug interactions using check_drug_interactions
4. Provide a clear summary of which trials the patient likely qualifies for, which they might not, and why

Be thorough but concise. Always explain your reasoning about eligibility.
Flag any potential concerns (age limits, medication conflicts, exclusion criteria).
If the patient might qualify, say so clearly. If they probably don't, explain exactly which criteria they fail.

IMPORTANT: Always return your final answer as a structured JSON object with this format:
{
    "matches": [
        {
            "nct_id": "NCT...",
            "title": "Trial title",
            "match_strength": "strong" | "moderate" | "weak",
            "explanation": "Why this trial matches or doesn't match",
            "concerns": ["any potential issues"],
            "next_steps": "what the patient should do"
        }
    ],
    "summary": "Overall summary of findings",
    "medications_checked": ["list of medications checked for interactions"]
}"""

    # Build the user message
    user_msg = f"Patient description: {patient.description}"
    if patient.age:
        user_msg += f"\nAge: {patient.age}"
    if patient.sex:
        user_msg += f"\nSex: {patient.sex}"
    if patient.medications:
        user_msg += f"\nCurrent medications: {', '.join(patient.medications)}"
    if patient.country:
        user_msg += f"\nCountry: {patient.country}"
    if patient.state:
        user_msg += f"\nState: {patient.state}"
    user_msg += f"\n\nPlease find up to {patient.max_results} matching clinical trials."

    messages = [{"role": "user", "content": user_msg}]

    # === AGENTIC LOOP ===
    # Claude might need to call multiple tools in sequence.
    # Each tool call returns data, which Claude uses to decide
    # what to do next. This loop continues until Claude gives
    # a final text response (no more tool calls).
    max_iterations = 10  # Safety limit to prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Claude wants to call one or more tools
            tool_results = []

            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    tool_id = content.id

                    # Route to the right function
                    result = _call_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    })

            # Add Claude's response and tool results to the conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude is done — extract the final text response
            final_text = ""
            for content in response.content:
                if hasattr(content, "text"):
                    final_text += content.text

            # Try to parse as JSON, fall back to raw text
            try:
                # Sometimes Claude wraps JSON in markdown code blocks
                cleaned = final_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return {"summary": final_text, "matches": [], "raw_response": True}

    return {"error": "Agent reached maximum iterations without completing", "matches": []}


def _call_tool(tool_name: str, tool_input: dict) -> str:
    """Route a tool call to the right function and return the result."""
    tool_map = {
        "search_trials": search_trials,
        "get_trial_details": get_trial_details,
        "get_eligibility_criteria": get_eligibility_criteria,
        "get_drug_info": get_drug_info,
        "check_drug_interactions": check_drug_interactions,
    }

    func = tool_map.get(tool_name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        return func(**tool_input)
    except Exception as e:
        return json.dumps({"error": f"Tool '{tool_name}' failed: {str(e)}"})


# === API ENDPOINTS ===

@app.get("/health")
def health_check():
    """Simple health check — is the server running?"""
    return {"status": "healthy", "version": "0.2.0"}


@app.post("/match")
def match_patient(patient: PatientInput):
    """
    Main endpoint: match a patient to clinical trials using AI.

    Send a patient description and get back AI-analyzed trial matches
    with eligibility explanations.
    """
    start_time = time.time()

    try:
        result = run_agent(patient)
        elapsed = round(time.time() - start_time, 2)
        result["processing_time_seconds"] = elapsed
        return result
    except anthropic.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid Anthropic API key. Check your .env file."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(query: SearchInput):
    """
    Direct search — no AI, just OpenSearch.
    Useful for quick searches without the Claude overhead.
    """
    results = hybrid_search(
        query_text=query.query,
        min_age=query.age,
        max_age=query.age,
        sex=query.sex,
        country=query.country,
        state=query.state,
        phase=query.phase,
        status="RECRUITING",
        page_size=query.max_results,
    )

    # Remove vectors from response (huge and useless in API output)
    for trial in results["trials"]:
        trial.pop("search_vector", None)
        trial.pop("search_text", None)

    return results


@app.get("/trial/{nct_id}")
def get_trial(nct_id: str):
    """Get full details for a specific trial."""
    result = get_trial_details(nct_id)
    parsed = json.loads(result)
    if "error" in parsed:
        raise HTTPException(status_code=404, detail=parsed["error"])
    return parsed