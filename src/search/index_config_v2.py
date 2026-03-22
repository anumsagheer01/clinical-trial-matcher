"""
Updated OpenSearch index config - now with vector search support.

What changed from v1:
- Added "search_vector" field (type: knn_vector, 384 dimensions)
- Added k-NN settings to enable the vector search engine
- Everything else stays the same

This means I can now do HYBRID search:
- Keyword search (the original way) - good for exact medical terms
- Vector search (new) - good for meaning-based matching
- Combined - best of both 
"""

INDEX_NAME = "clinical_trials"

INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            # This enables the k-NN plugin for vector search
            "knn": True,
        },
        "analysis": {
            "analyzer": {
                "medical_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "medical_synonyms"]
                }
            },
            "filter": {
                "medical_synonyms": {
                    "type": "synonym",
                    "synonyms": [
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
            # === THE NEW VECTOR FIELD ===
            "search_vector": {
                "type": "knn_vector",
                "dimension": 384,  # Must match the embedding model output size
                "method": {
                    "name": "hnsw",        # Algorithm for fast nearest-neighbor search
                    "space_type": "cosinesimil", # Measure similarity by angle, not distance
                    "engine": "nmslib",     # The underlying search library
                },
            },

            # === Everything below is same as before ===
            "nct_id": {"type": "keyword"},
            "brief_title": {
                "type": "text",
                "analyzer": "medical_analyzer",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "official_title": {"type": "text", "analyzer": "medical_analyzer"},
            "brief_summary": {"type": "text", "analyzer": "medical_analyzer"},
            "detailed_description": {"type": "text", "analyzer": "medical_analyzer"},
            "search_text": {"type": "text", "analyzer": "medical_analyzer"},
            "overall_status": {"type": "keyword"},
            "study_type": {"type": "keyword"},
            "phases": {"type": "keyword"},
            "lead_sponsor": {
                "type": "keyword",
                "fields": {"text": {"type": "text"}}
            },
            "conditions": {
                "type": "keyword",
                "fields": {"text": {"type": "text", "analyzer": "medical_analyzer"}}
            },
            "keywords": {
                "type": "keyword",
                "fields": {"text": {"type": "text", "analyzer": "medical_analyzer"}}
            },
            "interventions": {
                "type": "nested",
                "properties": {
                    "name": {"type": "text"},
                    "type": {"type": "keyword"},
                    "description": {"type": "text"},
                }
            },
            "eligibility": {
                "properties": {
                    "criteria_text": {"type": "text", "analyzer": "medical_analyzer"},
                    "min_age": {"type": "float"},
                    "max_age": {"type": "float"},
                    "sex": {"type": "keyword"},
                    "inclusion_criteria": {"type": "text"},
                    "exclusion_criteria": {"type": "text"},
                }
            },
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
            "enrollment_count": {"type": "integer"},
            "start_date": {"type": "keyword"},
            "completion_date": {"type": "keyword"},
            "last_update": {"type": "keyword"},
        }
    }
}