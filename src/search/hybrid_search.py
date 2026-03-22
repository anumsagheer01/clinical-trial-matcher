"""
Hybrid search combines keyword search and vector search.

Because keyword and vector search catch different things:

Keyword search is good at:
- Exact medical terms ("NCT04280705", "Phase 3", "metformin")
- Structured filters (age, sex, location)

Vector search is good at:
- Meaning ("my kidneys are failing" and  matches "renal insufficiency")
- Paraphrases ("trouble breathing" and matches "respiratory distress")
- Imprecise patient language and precise medical terminology

Hybrid = run both, combine the scores, get the best of both

"""

import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch

from src.search.index_config_v2 import INDEX_NAME
from src.search.embeddings import generate_embedding

load_dotenv()


def get_client():
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )


def hybrid_search(
    query_text=None,
    min_age=None,
    max_age=None,
    sex=None,
    country=None,
    state=None,
    phase=None,
    status=None,
    page_size=10,
    keyword_weight=0.5,
    vector_weight=0.5,
):
    """
    Run a hybrid keyword + vector search.

    The keyword_weight and vector_weight control how much each
    method contributes to the final score. Default is 50/50.

    I'll test different weight ratios
    to find which combo gives the best results for clinical trials.
    """
    client = get_client()

    # Build the filter clauses (same as before, these are hard filters)
    filter_clauses = []

    if min_age is not None:
        filter_clauses.append({
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "eligibility.min_age"}}}},
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

    if country or state:
        nested_must = []
        if country:
            nested_must.append({"term": {"locations.country": country}})
        if state:
            nested_must.append({"term": {"locations.state": state}})
        filter_clauses.append({
            "nested": {
                "path": "locations",
                "query": {"bool": {"must": nested_must}}
            }
        })

    if phase:
        filter_clauses.append({"term": {"phases": phase}})
    if status:
        filter_clauses.append({"term": {"overall_status": status}})

    # If no query text, just do a filtered match_all
    if not query_text:
        query = {"match_all": {}}
        if filter_clauses:
            query = {"bool": {"must": [{"match_all": {}}], "filter": filter_clauses}}

        response = client.search(
            index=INDEX_NAME,
            body={"query": query, "size": page_size}
        )
        return _parse_response(response, 1, page_size)

    # === HYBRID SEARCH STRATEGY ===
    # I run two separate searches and combine the results.
    # This is simpler and more controllable than OpenSearch's
    # built-in hybrid query (which is still experimental).

    # Search 1: Keyword search
    keyword_query = {
        "bool": {
            "must": [{
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "brief_title^3",
                        "conditions.text^2",
                        "keywords.text^2",
                        "brief_summary",
                        "search_text",
                        "eligibility.criteria_text",
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO" if len(query_text.split()) <= 4 else "0",
                }
            }],
        }
    }
    if filter_clauses:
        keyword_query["bool"]["filter"] = filter_clauses

    keyword_response = client.search(
        index=INDEX_NAME,
        body={"query": keyword_query, "size": page_size * 2}
        # Fetch extra results so I have a bigger pool to merge from
    )

    # Search 2: Vector search
    query_vector = generate_embedding(query_text)

    vector_query = {
        "knn": {
            "search_vector": {
                "vector": query_vector,
                "k": page_size * 2,  # How many nearest neighbors to find
            }
        }
    }

    # k-NN doesn't support bool filters directly in all OpenSearch versions,
    # so I apply filters after retrieval by fetching extra and filtering
    vector_response = client.search(
        index=INDEX_NAME,
        body={
            "query": vector_query,
            "size": page_size * 3,  # Fetch more to compensate for post-filtering
        }
    )

    # === COMBINE RESULTS ===
    # Reciprocal Rank Fusion (RRF) method for combining
    # ranked lists. The idea: a document ranked #1 gets more credit
    # than one ranked #10, regardless of the raw scores.
    #
    # Formula: RRF_score = sum(1 / (k + rank)) for each list
    # k=60 is standard (from the original RRF paper)

    k = 60  # RRF constant
    combined_scores = {}
    trial_data = {}

    # Score keyword results
    for rank, hit in enumerate(keyword_response["hits"]["hits"]):
        doc_id = hit["_id"]
        rrf_score = keyword_weight * (1.0 / (k + rank + 1))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
        trial_data[doc_id] = hit["_source"]

    # Score vector results
    for rank, hit in enumerate(vector_response["hits"]["hits"]):
        doc_id = hit["_id"]
        rrf_score = vector_weight * (1.0 / (k + rank + 1))
        combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
        if doc_id not in trial_data:
            trial_data[doc_id] = hit["_source"]

    # Sort by combined score (highest first)
    sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    # Apply filters to vector results that might not have been filtered
    # (This is the post-filtering step)
    filtered_trials = []
    for doc_id in sorted_ids:
        trial = trial_data[doc_id]

        # Check filters manually for vector results
        if not _passes_filters(trial, min_age, max_age, sex, country, state, phase, status):
            continue

        trial["_score"] = combined_scores[doc_id]
        filtered_trials.append(trial)

        if len(filtered_trials) >= page_size:
            break

    return {
        "total": len(combined_scores),
        "trials": filtered_trials,
        "took_ms": keyword_response["took"] + vector_response["took"],
        "search_type": "hybrid",
    }


def _passes_filters(trial, min_age, max_age, sex, country, state, phase, status):
    """Check if a trial passes all the given filters."""
    elig = trial.get("eligibility", {})

    if min_age is not None:
        trial_min = elig.get("min_age")
        if trial_min is not None and trial_min > min_age:
            return False

    if max_age is not None:
        trial_max = elig.get("max_age")
        if trial_max is not None and trial_max < max_age:
            return False

    if sex:
        trial_sex = elig.get("sex", "All")
        if trial_sex not in ["All", sex]:
            return False

    if country:
        locations = trial.get("locations", [])
        if locations and not any(loc.get("country") == country for loc in locations):
            return False

    if state:
        locations = trial.get("locations", [])
        if locations and not any(loc.get("state") == state for loc in locations):
            return False

    if phase:
        if phase not in trial.get("phases", []):
            return False

    if status:
        if trial.get("overall_status") != status:
            return False

    return True


def _parse_response(response, page, page_size):
    """Parse a standard OpenSearch response into my format."""
    hits = response["hits"]
    trials = []
    for hit in hits["hits"]:
        trial = hit["_source"]
        trial["_score"] = hit.get("_score", 0)
        trials.append(trial)
    return {
        "total": hits["total"]["value"],
        "trials": trials,
        "took_ms": response["took"],
        "search_type": "standard",
    }


# === DEMO ===
if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID SEARCH DEMO")
    print("=" * 60)

    # Test 1: Patient-style natural language
    print("\n>>> 'my kidneys are failing and I have high blood pressure'")
    results = hybrid_search(
        query_text="my kidneys are failing and I have high blood pressure",
        page_size=5,
    )
    print(f"Found {results['total']} results in {results['took_ms']}ms")
    for i, trial in enumerate(results["trials"], 1):
        print(f"  {i}. [{trial['nct_id']}] {trial['brief_title']}")
        print(f"     Conditions: {', '.join(trial['conditions'][:3])}")

    # Test 2: Exact medical term
    print("\n>>> 'metformin phase 3 type 2 diabetes' for 55M in US")
    results = hybrid_search(
        query_text="metformin phase 3 type 2 diabetes",
        min_age=55, max_age=55, sex="Male",
        country="United States",
        page_size=5,
    )
    print(f"Found {results['total']} results in {results['took_ms']}ms")
    for i, trial in enumerate(results["trials"], 1):
        print(f"  {i}. [{trial['nct_id']}] {trial['brief_title']}")