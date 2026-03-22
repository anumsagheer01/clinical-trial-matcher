"""
Search clinical trials in OpenSearch.

This module provides the search functions that the rest of the
application will use. It supports:
1. Text search (find trials by condition or keyword)
2. Filtered search (narrow by age, sex, location, status, phase)
3. Combined search (text + filters together)

SEARCH STRATEGY:
Use OpenSearch's "bool" query, which lets combine:
- "must"  Results MUST match these (like AND)
- "should"  Results SHOULD match these (like OR, boosts relevance)
- "filter"  Hard filters (yes/no, no relevance scoring)

This combination is how every major search engine works.
"""

import os
import json
from dotenv import load_dotenv
from opensearchpy import OpenSearch

from src.search.index_config import INDEX_NAME

load_dotenv()


def get_client():
    """Get an OpenSearch client."""
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )


def search_trials(
    query_text=None,
    conditions=None,
    min_age=None,
    max_age=None,
    sex=None,
    country=None,
    state=None,
    city=None,
    phase=None,
    status=None,
    page_size=10,
    page=1,
):
    """
    Search for clinical trials with text and/or filters.
    
    Args:
        query_text: Free-text search (e.g., "diabetes kidney disease")
        conditions: List of conditions to filter by
        min_age: Patient's age - filter for trials that accept this age
        max_age: Patient's age - same as above (usually same as min_age)
        sex: "Male", "Female", or None for any
        country: Country filter (e.g., "United States")
        state: State filter (e.g., "Maryland")
        city: City filter
        phase: Trial phase filter (e.g., "PHASE3")
        status: Trial status (e.g., "RECRUITING")
        page_size: Number of results per page
        page: Page number (1-indexed)
    
    Returns:
        dict with "total" (count), "trials" (list), and "took_ms" (search time)
    """
    client = get_client()
    
    # Build the query
    must_clauses = []      # Results MUST match all of these
    should_clauses = []    # Results SHOULD match (boosts relevance)
    filter_clauses = []    # Hard filters (no relevance impact)
    
    # --- TEXT SEARCH ---
    if query_text:
        must_clauses.append({
            "multi_match": {
                "query": query_text,
                "fields": [
                    "brief_title^3",        # Title matches are 3x more important
                    "conditions.text^2",     # Condition matches are 2x more important
                    "keywords.text^2",       # Keywords are 2x important
                    "brief_summary",         # Summary matches at normal weight
                    "search_text",           # Catch-all field
                    "eligibility.criteria_text",
                ],
                "type": "best_fields",     # Use the best-matching field's score
                "fuzziness": "AUTO",       # Handle typos: "diabets" to "diabetes"
            }
        })
    
    # --- CONDITION FILTER ---
    if conditions:
        for condition in conditions:
            should_clauses.append({
                "match": {
                    "conditions.text": {
                        "query": condition,
                        "boost": 2,
                    }
                }
            })
    
    # --- AGE FILTER ---
    # A trial is eligible if:
    #   trial's min_age <= patient's age AND trial's max_age >= patient's age
    if min_age is not None:
        filter_clauses.append({
            "bool": {
                "should": [
                    # Trial has no min age (accepts any age)
                    {"bool": {"must_not": {"exists": {"field": "eligibility.min_age"}}}},
                    # Trial's min age is <= patient's age
                    {"range": {"eligibility.min_age": {"lte": min_age}}},
                ],
                "minimum_should_match": 1,
            }
        })
    
    if max_age is not None:
        filter_clauses.append({
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "eligibility.max_age"}}}},
                    {"range": {"eligibility.max_age": {"gte": max_age}}},
                ],
                "minimum_should_match": 1,
            }
        })
    
    # --- SEX FILTER ---
    if sex:
        filter_clauses.append({
            "bool": {
                "should": [
                    {"term": {"eligibility.sex": "All"}},
                    {"term": {"eligibility.sex": sex}},
                ],
                "minimum_should_match": 1,
            }
        })
    
    # --- LOCATION FILTER ---
    location_filters = {}
    if country:
        location_filters["locations.country"] = country
    if state:
        location_filters["locations.state"] = state
    if city:
        location_filters["locations.city"] = city
    
    if location_filters:
        nested_must = [{"term": {k: v}} for k, v in location_filters.items()]
        filter_clauses.append({
            "nested": {
                "path": "locations",
                "query": {
                    "bool": {
                        "must": nested_must
                    }
                }
            }
        })
    
    # --- PHASE FILTER ---
    if phase:
        filter_clauses.append({"term": {"phases": phase}})
    
    # --- STATUS FILTER ---
    if status:
        filter_clauses.append({"term": {"overall_status": status}})
    
    # --- BUILD FINAL QUERY ---
    bool_query = {}
    if must_clauses:
        bool_query["must"] = must_clauses
    if should_clauses:
        bool_query["should"] = should_clauses
    if filter_clauses:
        bool_query["filter"] = filter_clauses
    
    # If no criteria at all, match everything
    if not bool_query:
        query = {"match_all": {}}
    else:
        query = {"bool": bool_query}
    
    # Calculate pagination offset
    from_offset = (page - 1) * page_size
    
    # Execute the search
    response = client.search(
        index=INDEX_NAME,
        body={
            "query": query,
            "size": page_size,
            "from": from_offset,
            "sort": [
                "_score",                          # Most relevant first
                {"last_update": {"order": "desc"}}, # Then most recently updated
            ],
            # Highlight matching text (shows WHY a result matched)
            "highlight": {
                "fields": {
                    "brief_title": {},
                    "brief_summary": {"fragment_size": 200},
                    "conditions.text": {},
                }
            },
        },
    )
    
    # Parse the response
    hits = response["hits"]
    total = hits["total"]["value"]
    took_ms = response["took"]
    
    trials = []
    for hit in hits["hits"]:
        trial = hit["_source"]
        trial["_score"] = hit["_score"]
        trial["_highlights"] = hit.get("highlight", {})
        trials.append(trial)
    
    return {
        "total": total,
        "trials": trials,
        "took_ms": took_ms,
        "page": page,
        "page_size": page_size,
    }


def print_search_results(results):
    """Pretty-print search results for testing."""
    print(f"\n  Found {results['total']} trials in {results['took_ms']}ms")
    print(f"  Showing page {results['page']} ({len(results['trials'])} results)")
    print("-" * 70)
    
    for i, trial in enumerate(results["trials"], 1):
        score = trial.get("_score", 0)
        print(f"\n  {i}. [{trial['nct_id']}] (score: {score:.2f})")
        print(f"     {trial['brief_title']}")
        print(f"     Status: {trial['overall_status']} | Phase: {', '.join(trial['phases']) if trial['phases'] else 'N/A'}")
        print(f"     Conditions: {', '.join(trial['conditions'][:3])}")
        
        # Show eligibility
        elig = trial.get("eligibility", {})
        age_str = ""
        if elig.get("min_age"):
            age_str += f"Min: {elig['min_age']}y"
        if elig.get("max_age"):
            age_str += f" Max: {elig['max_age']}y"
        if age_str:
            print(f"     Age: {age_str} | Sex: {elig.get('sex', 'All')}")
        
        # Show location count
        loc_count = len(trial.get("locations", []))
        if loc_count > 0:
            first_loc = trial["locations"][0]
            loc_str = f"{first_loc.get('city', '')}, {first_loc.get('state', '')}, {first_loc.get('country', '')}"
            print(f"     Locations: {loc_count} sites (first: {loc_str})")
        
        # Show highlights
        highlights = trial.get("_highlights", {})
        if highlights:
            for field, fragments in highlights.items():
                print(f"     Matched in {field}: ...{fragments[0]}...")


# === DEMO SEARCHES ===


if __name__ == "__main__":
    print("=" * 70)
    print("CLINICAL TRIAL SEARCH- DEMO QUERIES")
    print("=" * 70)
    
    # Search 1: Simple text search
    print("\n\n>>> SEARCH 1: 'type 2 diabetes'")
    results = search_trials(query_text="type 2 diabetes", page_size=5)
    print_search_results(results)
    
    # Search 2: With age and sex filter
    print("\n\n>>> SEARCH 2: 'breast cancer' for 45-year-old female")
    results = search_trials(
        query_text="breast cancer",
        min_age=45,
        max_age=45,
        sex="Female",
        page_size=5,
    )
    print_search_results(results)
    
    # Search 3: Location-specific
    print("\n\n>>> SEARCH 3: 'depression' trials in Maryland")
    results = search_trials(
        query_text="depression",
        state="Maryland",
        country="United States",
        page_size=5,
    )
    print_search_results(results)
    
    # Search 4: Phase 3 recruiting trials for a specific condition
    print("\n\n>>> SEARCH 4: Phase 3 recruiting 'lung cancer' trials")
    results = search_trials(
        query_text="lung cancer",
        phase="PHASE3",
        status="RECRUITING",
        page_size=5,
    )
    print_search_results(results)
    
    # Search 5: Complex patient query
    print("\n\n>>> SEARCH 5: Complex query - 'chronic kidney disease hypertension' for 62M in United States")
    results = search_trials(
        query_text="chronic kidney disease hypertension",
        min_age=62,
        max_age=62,
        sex="Male",
        country="United States",
        page_size=5,
    )
    print_search_results(results)