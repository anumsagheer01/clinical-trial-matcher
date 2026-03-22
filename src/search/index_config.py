"""
OpenSearch index configuration for clinical trials.

This defines the MAPPING, the schema that tells OpenSearch
what each field is and how to search it.

FIELD TYPES EXPLAINED:
- "text"    → Full-text searchable. "Type 2 Diabetes" matches "diabetes".
              OpenSearch breaks it into tokens and indexes each word.
- "keyword" → Exact match only. "RECRUITING" only matches "RECRUITING".
              Used for filtering, sorting, aggregations.
- "integer" → Numeric. Supports range queries like "age >= 18".
- "float"   → Decimal number. Same as integer but with decimals.
- "date"    → Date values. Supports range queries.
- "nested"  → An array of objects where each object is independently searchable.
              Without "nested", OpenSearch flattens arrays and cross-matches fields.
- "boolean" → True/false.

WHY THIS MATTERS:
If you index "New York" as "text", searching for "york" will match it.
If you index "New York" as "keyword", only "New York" (exact) matches.
Choosing the right type for each field is critical for search quality.
"""

# The name of our index in OpenSearch
INDEX_NAME = "clinical_trials"

# The mapping (schema) definition
INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,        # 1 shard is fine for our data size
            "number_of_replicas": 0,       # 0 replicas for local dev (no redundancy needed)
        },
        "analysis": {
            "analyzer": {
                "medical_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",       # Splits on whitespace and punctuation
                    "filter": [
                        "lowercase",               # "Diabetes" - "diabetes"
                        "medical_synonyms",         # "T2DM" - "type 2 diabetes mellitus"
                    ]
                }
            },
            "filter": {
                "medical_synonyms": {
                    "type": "synonym",
                    "synonyms": [
                        # Common medical abbreviations
                        # These help match when patients use informal terms
                        "t2dm, type 2 diabetes mellitus, type 2 diabetes",
                        "t1dm, type 1 diabetes mellitus, type 1 diabetes",
                        "ckd, chronic kidney disease",
                        "copd, chronic obstructive pulmonary disease",
                        "chf, congestive heart failure",
                        "htn, hypertension, high blood pressure",
                        "mi, myocardial infarction, heart attack",
                        "cad, coronary artery disease",
                        "hiv, human immunodeficiency virus",
                        "nsclc, non small cell lung cancer",
                        "sclc, small cell lung cancer",
                        "ra, rheumatoid arthritis",
                        "ms, multiple sclerosis",
                        "ibd, inflammatory bowel disease",
                        "uc, ulcerative colitis",
                        "cd, crohn disease, crohns disease",
                        "adhd, attention deficit hyperactivity disorder",
                        "ptsd, post traumatic stress disorder",
                        "mdd, major depressive disorder, major depression",
                        "aki, acute kidney injury",
                        "ards, acute respiratory distress syndrome",
                        "bph, benign prostatic hyperplasia",
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # === IDENTIFICATION ===
            "nct_id": {
                "type": "keyword"  # Exact match - NCT IDs are unique identifiers
            },
            
            # === TEXT FIELDS (full-text searchable) ===
            "brief_title": {
                "type": "text",
                "analyzer": "medical_analyzer",
                # Also store as keyword for exact matching and sorting
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "official_title": {
                "type": "text",
                "analyzer": "medical_analyzer"
            },
            "brief_summary": {
                "type": "text",
                "analyzer": "medical_analyzer"
            },
            "detailed_description": {
                "type": "text",
                "analyzer": "medical_analyzer"
            },
            "search_text": {
                "type": "text",
                "analyzer": "medical_analyzer"
            },
            
            # === CATEGORICAL FIELDS (exact match / filtering) ===
            "overall_status": {
                "type": "keyword"
            },
            "study_type": {
                "type": "keyword"
            },
            "phases": {
                "type": "keyword"  # Array of keywords- OpenSearch handles arrays natively
            },
            "lead_sponsor": {
                "type": "keyword",
                "fields": {
                    "text": {"type": "text"}  # Also searchable as text
                }
            },
            
            # === MEDICAL FIELDS ===
            "conditions": {
                "type": "keyword",
                "fields": {
                    "text": {
                        "type": "text",
                        "analyzer": "medical_analyzer"
                    }
                }
            },
            "keywords": {
                "type": "keyword",
                "fields": {
                    "text": {
                        "type": "text",
                        "analyzer": "medical_analyzer"
                    }
                }
            },
            
            # === INTERVENTIONS (nested — each intervention is independent) ===
            "interventions": {
                "type": "nested",
                "properties": {
                    "name": {"type": "text"},
                    "type": {"type": "keyword"},
                    "description": {"type": "text"},
                }
            },
            
            # === ELIGIBILITY ===
            "eligibility": {
                "properties": {
                    "criteria_text": {
                        "type": "text",
                        "analyzer": "medical_analyzer"
                    },
                    "min_age": {"type": "float"},
                    "max_age": {"type": "float"},
                    "sex": {"type": "keyword"},
                    "inclusion_criteria": {"type": "text"},
                    "exclusion_criteria": {"type": "text"},
                }
            },
            
            # === LOCATIONS (nested) ===
            "locations": {
                "type": "nested",
                "properties": {
                    "facility": {"type": "text"},
                    "city": {"type": "keyword"},
                    "state": {"type": "keyword"},
                    "country": {"type": "keyword"},
                    "status": {"type": "keyword"},
                }
            },
            
            # === NUMERIC ===
            "enrollment_count": {
                "type": "integer"
            },
            
            # === DATES ===
            "start_date": {
                "type": "keyword"  
            },
            "completion_date": {
                "type": "keyword"
            },
            "last_update": {
                "type": "keyword"
            },
        }
    }
}