"""
Parse raw ClinicalTrials.gov data into clean, structured format.

The raw API response has deeply nested JSON. This script:
1. Reads the raw data
2. Extracts the fields we care about
3. Handles missing data gracefully
4. Outputs clean JSON ready for OpenSearch indexing

WHY THIS STEP MATTERS:
- Raw data has 50+ nested fields; we need ~20
- Some fields are missing on certain trials
- Dates are in various formats
- Eligibility criteria text needs cleaning
- OpenSearch needs a flat(ish) structure to index efficiently
"""

import json
import os
import re
from datetime import datetime
from tqdm import tqdm


RAW_DATA_PATH = os.path.join("data", "raw", "raw_trials.json")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "trials_cleaned.json")


def safe_get(data, *keys, default=None):
    """
    Safely navigate nested dictionaries.
    
    Instead of writing:
        data.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
    
    Write:
        safe_get(data, "protocolSection", "identificationModule", "nctId", default="")
    
    If any key is missing, returns the default instead of crashing.
    """
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current


def parse_eligibility_criteria(eligibility_module):
    """
    Parse the eligibility criteria from a trial.
    
    The API returns eligibility as a block of text like:
        "Inclusion Criteria:
         - Must be 18 years or older
         - Diagnosed with Type 2 Diabetes
         Exclusion Criteria:
         - Pregnant or breastfeeding
         - Active cancer"
    
    We extract:
    - The raw text (for semantic search later)
    - Min/max age (for filtering)
    - Sex (for filtering)
    - Structured inclusion/exclusion lists (for display)
    """
    if not eligibility_module:
        return {
            "criteria_text": "",
            "min_age": None,
            "max_age": None,
            "sex": "All",
            "inclusion_criteria": [],
            "exclusion_criteria": [],
        }
    
    # Get structured fields
    min_age_str = eligibility_module.get("minimumAge", "")
    max_age_str = eligibility_module.get("maximumAge", "")
    sex = eligibility_module.get("sex", "All")
    criteria_text = eligibility_module.get("eligibilityCriteria", "")
    
    # Parse age strings like "18 Years" into numbers
    min_age = parse_age(min_age_str)
    max_age = parse_age(max_age_str)
    
    # Split criteria into inclusion and exclusion lists
    inclusion, exclusion = split_criteria(criteria_text)
    
    return {
        "criteria_text": criteria_text,
        "min_age": min_age,
        "max_age": max_age,
        "sex": sex,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
    }


def parse_age(age_str):
    """
    Convert age strings like '18 Years', '6 Months' into a number (in years).
    
    Examples:
        '18 Years' → 18
        '6 Months' → 0.5
        '' → None
    """
    if not age_str:
        return None
    
    match = re.match(r"(\d+)\s*(Year|Month|Week|Day)", age_str, re.IGNORECASE)
    if not match:
        return None
    
    value = int(match.group(1))
    unit = match.group(2).lower()
    
    if unit == "year":
        return value
    elif unit == "month":
        return round(value / 12, 1)
    elif unit == "week":
        return round(value / 52, 1)
    elif unit == "day":
        return round(value / 365, 1)
    
    return None


def split_criteria(criteria_text):
    """
    Split eligibility criteria text into inclusion and exclusion lists.
    
    The text typically has sections like:
        Inclusion Criteria:
        - criterion 1
        - criterion 2
        
        Exclusion Criteria:
        - criterion 3
    """
    inclusion = []
    exclusion = []
    
    if not criteria_text:
        return inclusion, exclusion
    
    # Find the inclusion and exclusion sections
    text = criteria_text.strip()
    
    # Try to split on "Exclusion Criteria" (case-insensitive)
    parts = re.split(r"(?i)exclusion\s+criteria\s*:?", text)
    
    inclusion_text = parts[0] if len(parts) > 0 else ""
    exclusion_text = parts[1] if len(parts) > 1 else ""
    
    # Remove "Inclusion Criteria:" header from inclusion text
    inclusion_text = re.sub(r"(?i)inclusion\s+criteria\s*:?", "", inclusion_text)
    
    # Split each section into individual criteria
    inclusion = extract_criteria_list(inclusion_text)
    exclusion = extract_criteria_list(exclusion_text)
    
    return inclusion, exclusion


def extract_criteria_list(text):
    """
    Extract individual criteria from a text block.
    Handles bullet points, numbered lists, and newline-separated items.
    """
    if not text.strip():
        return []
    
    # Split on common list markers
    items = re.split(r"\n\s*[-•*]\s*|\n\s*\d+[.)]\s*|\n{2,}", text)
    
    # Clean up each item
    criteria = []
    for item in items:
        cleaned = item.strip()
        # Remove leading bullets/numbers if regex didn't catch them
        cleaned = re.sub(r"^[-•*]\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        # Only keep items with actual content (at least 10 chars)
        if len(cleaned) >= 10:
            criteria.append(cleaned)
    
    return criteria


def parse_locations(protocol_section):
    """
    Extract trial locations (where patients can go to participate).
    
    Returns a list of location objects with facility, city, state, country.
    """
    locations_module = safe_get(
        protocol_section, "contactsLocationsModule", default={}
    )
    raw_locations = locations_module.get("locations", [])
    
    parsed_locations = []
    for loc in raw_locations:
        parsed_locations.append({
            "facility": loc.get("facility", ""),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
            "status": loc.get("status", ""),
        })
    
    return parsed_locations


def parse_single_trial(study):
    """
    Parse one trial from the raw API format into our clean format.
    
    This is the core transformation function. It takes the messy nested
    API response for a single trial and returns a flat, clean dictionary.
    """
    protocol = study.get("protocolSection", {})
    
    # === IDENTIFICATION ===
    id_module = protocol.get("identificationModule", {})
    nct_id = id_module.get("nctId", "")
    brief_title = id_module.get("briefTitle", "")
    official_title = id_module.get("officialTitle", "")
    
    # === DESCRIPTION ===
    desc_module = protocol.get("descriptionModule", {})
    brief_summary = desc_module.get("briefSummary", "")
    detailed_description = desc_module.get("detailedDescription", "")
    
    # === STATUS & DESIGN ===
    status_module = protocol.get("statusModule", {})
    overall_status = status_module.get("overallStatus", "")
    start_date = safe_get(status_module, "startDateStruct", "date", default="")
    completion_date = safe_get(status_module, "completionDateStruct", "date", default="")
    last_update = safe_get(status_module, "lastUpdatePostDateStruct", "date", default="")
    
    design_module = protocol.get("designModule", {})
    study_type = design_module.get("studyType", "")
    phases = design_module.get("phases", [])
    enrollment_info = design_module.get("enrollmentInfo", {})
    enrollment_count = enrollment_info.get("count", 0)
    
    # === CONDITIONS & KEYWORDS ===
    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])
    keywords = conditions_module.get("keywords", [])
    
    # === INTERVENTIONS ===
    arms_module = protocol.get("armsInterventionsModule", {})
    raw_interventions = arms_module.get("interventions", [])
    interventions = []
    for interv in raw_interventions:
        interventions.append({
            "name": interv.get("name", ""),
            "type": interv.get("type", ""),
            "description": interv.get("description", ""),
        })
    
    # === ELIGIBILITY ===
    eligibility_module = protocol.get("eligibilityModule", {})
    eligibility = parse_eligibility_criteria(eligibility_module)
    
    # === LOCATIONS ===
    locations = parse_locations(protocol)
    
    # === SPONSOR ===
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor = safe_get(sponsor_module, "leadSponsor", "name", default="")
    
    # === BUILD THE CLEAN DOCUMENT ===
    
    # Create a combined text field for semantic search
    # This is what OpenSearch will use for vector search later
    search_text = " ".join(filter(None, [
        brief_title,
        brief_summary,
        " ".join(conditions),
        " ".join(keywords),
        eligibility.get("criteria_text", ""),
    ]))
    
    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,
        "overall_status": overall_status,
        "study_type": study_type,
        "phases": phases,
        "enrollment_count": enrollment_count,
        "conditions": conditions,
        "keywords": keywords,
        "interventions": interventions,
        "eligibility": eligibility,
        "locations": locations,
        "lead_sponsor": lead_sponsor,
        "start_date": start_date,
        "completion_date": completion_date,
        "last_update": last_update,
        "search_text": search_text,
    }


def parse_all_trials():
    """
    Parse all trials from the raw data file.
    
    Reads raw_trials.json, parses each trial, saves to trials_cleaned.json.
    Also prints statistics about the parsed data.
    """
    print("=" * 60)
    print("PARSING CLINICAL TRIAL DATA")
    print("=" * 60)
    
    # Load raw data
    print(f"\n  Loading raw data from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_studies = json.load(f)
    print(f"  Loaded {len(raw_studies)} raw trials")
    
    # Parse each trial
    print("\n  Parsing trials...")
    parsed_trials = []
    errors = 0
    
    for study in tqdm(raw_studies, desc="  Parsing"):
        try:
            parsed = parse_single_trial(study)
            parsed_trials.append(parsed)
        except Exception as e:
            errors += 1
            nct_id = safe_get(study, "protocolSection", "identificationModule", "nctId", default="UNKNOWN")
            if errors <= 5:  # Only print first 5 errors to avoid flooding
                print(f"\n  WARNING: Failed to parse {nct_id}: {e}")
    
    # Save parsed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    print(f"\n  Saving {len(parsed_trials)} parsed trials to {PROCESSED_DATA_PATH}...")
    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(parsed_trials, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("PARSING STATISTICS")
    print("=" * 60)
    print(f"  Total trials parsed: {len(parsed_trials)}")
    print(f"  Parse errors: {errors}")
    
    # Count trials with locations
    with_locations = sum(1 for t in parsed_trials if len(t["locations"]) > 0)
    print(f"  Trials with locations: {with_locations}")
    
    # Count unique conditions
    all_conditions = set()
    for t in parsed_trials:
        all_conditions.update(t["conditions"])
    print(f"  Unique conditions: {len(all_conditions)}")
    
    # Status breakdown
    status_counts = {}
    for t in parsed_trials:
        status = t["overall_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"  Status breakdown: {status_counts}")
    
    # Phase breakdown
    phase_counts = {}
    for t in parsed_trials:
        for phase in t["phases"]:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
    print(f"  Phase breakdown: {phase_counts}")
    
    return parsed_trials


if __name__ == "__main__":
    parse_all_trials()