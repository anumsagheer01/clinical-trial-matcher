# Clinical Trial Matcher

**Finding the right clinical trial should not take hours of research. This tool does it in seconds.**

When a drug company develops a new medicine, they need real patients to test it through clinical trials. There are over 92,000 trials actively recruiting patients right now, but most patients never find them because the search process is overwhelming. This platform changes that.

A patient types a simple description of their medical situation. The system searches 92,000+ trials, checks eligibility requirements, verifies drug interactions, and returns the best matches with clear explanations.

**Live Demo:** [clinical-trial-matcher.streamlit.app](https://clinical-trial-matcher.streamlit.app)
Note: This live demo uses a lightweight dataset of 500 trials. The full local version searches 92,746 trials with hybrid AI-powered retrieval.


## How It Works

A patient types something like:

> "I am a 58 year old male with type 2 diabetes and chronic kidney disease stage 3. I take metformin and lisinopril. I live in Maryland."

The system then:

1. **Extracts medical entities** from the text using a fine-tuned Flan-T5 model (age, sex, conditions, medications, location)
2. **Classifies query complexity** using a TensorFlow classifier to decide if the query needs simple search or full AI analysis
3. **Searches 92,746 clinical trials** using OpenSearch with hybrid retrieval (keyword search + vector search combined with Reciprocal Rank Fusion)
4. **Checks drug safety** by querying the FDA database through the openFDA API
5. **Reasons about eligibility** using Claude with multi-step tool calling
6. **Returns ranked results** with match explanations and direct links to ClinicalTrials.gov


## Architecture
```
Patient Input (free text)
        |
        v
[Entity Extraction Model]
  Fine-tuned Flan-T5-Small with PEFT/LoRA
  99% valid JSON output, 97.5% age accuracy
  30x cheaper than LLM-based extraction
        |
        v
[Query Complexity Classifier]
  TensorFlow binary classifier
  Routes simple queries to search only (skips LLM, saves cost)
  Routes complex queries to full AI pipeline
        |
        v
[Hybrid Search Engine]
  OpenSearch 2.11 with medical synonym analyzer
  Keyword search + k-NN vector search (384-dim embeddings)
  Combined using Reciprocal Rank Fusion
  92,746 trials indexed from ClinicalTrials.gov
        |
        v
[MCP Tool Servers]
  3 servers exposing 7 tools via Model Context Protocol
  Server 1: Trial search, details, eligibility, count
  Server 2: Drug info, interactions, warnings (openFDA)
  Server 3: Entity extraction (fine-tuned model)
        |
        v
[Claude Agent]
  Multi-step tool calling via Anthropic API
  Searches trials, checks eligibility, verifies drug safety
  Returns structured results with explanations
        |
        v
[Results]
  Ranked clinical trials with match strength
  Eligibility explanations
  Direct links to ClinicalTrials.gov
```


## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Search Engine | OpenSearch 2.11 | Hybrid keyword + vector search with medical synonyms |
| Embeddings | all-MiniLM-L6-v2 | 384-dimensional sentence embeddings for semantic search |
| Agent Protocol | MCP (Model Context Protocol) | Standardized tool interface for AI agents |
| LLM Reasoning | Claude via Anthropic API | Eligibility analysis with multi-step tool calling |
| Entity Extraction | Flan-T5-Small + PEFT/LoRA | Extracts structured patient data from free text |
| Query Routing | TensorFlow | Classifies queries to skip expensive LLM calls when possible |
| Backend | FastAPI | REST API with automatic documentation |
| Frontend | Streamlit | Accessible patient-facing interface |
| Containers | Docker + Docker Compose | 4-service stack in one command |
| CI/CD | GitHub Actions | Automated linting and Docker builds on every push |
| Orchestration | Kubernetes (manifests) | EKS-ready deployment with health probes and scaling |
| Drug Data | openFDA API | FDA label info, drug interactions, warnings |
| Trial Data | ClinicalTrials.gov API | 92,746 actively recruiting clinical trials |
| Evaluation | A/B Testing Framework | Compares retrieval strategies across 30 test queries |
| Testing | pytest | 30 tests covering search, API, models, and evaluation |


## Key Metrics

| What | Number |
|------|--------|
| Clinical trials indexed | 92,746 |
| Hybrid search latency | ~64ms |
| Entity extraction valid JSON rate | 99.0% |
| Entity extraction age accuracy | 97.5% |
| Entity extraction cost per query | ~$0.0001 (30x cheaper than LLM) |
| MCP tool servers | 3 servers, 7 tools |
| A/B test queries | 30 (keyword-heavy vs vector-heavy) |
| Automated tests | 30 passing |
| Docker services | 4 (OpenSearch, Dashboards, API, Frontend) |


## Project Structure
```
clinical-trial-matcher/
|
|-- .github/workflows/       CI/CD pipeline (GitHub Actions)
|-- k8s/                     Kubernetes deployment manifests
|   |-- namespace.yml        Namespace isolation
|   |-- opensearch-deployment.yml
|   |-- api-deployment.yml   API with health probes and 2 replicas
|   |-- frontend-deployment.yml
|   |-- ingress.yml          URL routing
|
|-- src/
|   |-- api/                 FastAPI backend with Claude agent loop
|   |-- entity_extraction/   Synthetic data generation + model training
|   |-- evaluation/          A/B testing framework
|   |-- ingestion/           Data download and parsing from ClinicalTrials.gov
|   |-- mcp_servers/         3 MCP tool servers
|   |   |-- trial_search_server.py    4 tools for trial search
|   |   |-- drug_info_server.py       3 tools for FDA drug data
|   |   |-- entity_extraction_server.py  1 tool for entity extraction
|   |-- query_classifier/    TensorFlow query complexity classifier
|   |-- search/              OpenSearch indexing, hybrid search, embeddings
|
|-- tests/                   30 tests (search, API, entities, classifier, A/B)
|-- models/                  Trained model artifacts (LoRA adapter, TF classifier)
|-- data/                    Raw and processed clinical trial data
|-- streamlit_app.py         Patient-facing frontend
|-- Dockerfile               API container
|-- Dockerfile.streamlit     Frontend container
|-- docker-compose.yml       Full stack orchestration (4 services)
|-- requirements.txt         Python dependencies
|-- demo_trials.json         500-trial subset for cloud demo
```


## Quick Start

### What You Need
- Python 3.11 or newer
- Docker Desktop
- Git

### Setup
```bash
git clone https://github.com/anumsagheer01/clinical-trial-matcher.git
cd clinical-trial-matcher

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Mac or Linux

pip install -r requirements_full.txt

docker compose up -d opensearch

python -m src.ingestion.download_trials
python -m src.ingestion.parse_trials
python scripts/reindex_with_vectors.py

python -m src.entity_extraction.generate_training_data
python -m src.entity_extraction.train_model
python -m src.query_classifier.classifier

streamlit run streamlit_app.py
```

### Run Tests
```bash
python -m pytest tests/ -v
```

### Run Everything in Docker
```bash
docker compose up -d --build
```
- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8501
- OpenSearch: http://localhost:9200

### Run the A/B Test
```bash
python -m src.evaluation.ab_test
```
## MCP Tool Servers

This project uses the Model Context Protocol (MCP) to expose backend capabilities as standardized tools that any AI agent can discover and call.

### Server 1: Clinical Trial Search (4 tools)

| Tool | What It Does |
|------|-------------|
| search_trials | Searches trials by condition, age, sex, location, and phase using hybrid retrieval |
| get_trial_details | Returns complete information for a specific trial by its NCT ID |
| get_eligibility_criteria | Returns inclusion and exclusion criteria for a specific trial |
| count_trials | Counts how many recruiting trials match given criteria |

### Server 2: Drug Information via openFDA (3 tools)

| Tool | What It Does |
|------|-------------|
| get_drug_info | Returns FDA label info including generic name, purpose, and dosage |
| check_drug_interactions | Checks for known drug interactions and contraindications |
| get_drug_warnings | Returns boxed warnings and adverse reactions |

### Server 3: Entity Extraction (1 tool)

| Tool | What It Does |
|------|-------------|
| extract_patient_entities | Extracts age, sex, conditions, medications, and location from patient text |


## A/B Testing

The retrieval system was evaluated using an A/B test comparing two strategies across 30 test queries:

- **Strategy A (keyword-heavy):** 70% keyword search weight, 30% vector search weight
- **Strategy B (vector-heavy):** 30% keyword search weight, 70% vector search weight

Metrics measured: retrieval relevance, search latency, result diversity, and coverage (percentage of queries returning 5+ results).

Results are saved in `results/ab_test_results.json` and were used to select the optimal retrieval configuration.



## How the Hybrid Search Works

Most search engines use either keyword matching or vector (semantic) search. Each has weaknesses:

- **Keyword search** misses synonyms. A patient searching "kidney failure" would not find trials about "renal insufficiency" because the words are different.
- **Vector search** misses exact terms. Searching for a specific trial ID like "NCT04280705" or a drug name like "metformin" may not match well semantically.

This system uses both, combined with Reciprocal Rank Fusion (RRF). Each search method produces a ranked list. RRF merges them by giving credit based on rank position, not raw scores. A trial ranked highly by both methods gets the top spot.

The OpenSearch index also includes a medical synonym analyzer that maps abbreviations to full terms (for example, "CKD" maps to "chronic kidney disease" and "HTN" maps to "hypertension").


## How the Entity Extraction Model Works

Instead of using Claude (slow and expensive) to parse every patient description, a small fine-tuned model handles this step.

- **Base model:** Google Flan-T5-Small (80M parameters)
- **Fine-tuning method:** PEFT with LoRA (rank 16), training only 0.5% of parameters
- **Training data:** 2,000 synthetic patient descriptions generated with a rule-based system covering 40+ medical conditions, 30+ medications, and 15+ US locations
- **Output:** Structured JSON with age, sex, conditions, medications, and location
- **Result:** 99% valid JSON output rate, 97.5% age extraction accuracy, at 30x lower cost than calling an LLM


## Kubernetes Deployment

The `k8s/` directory contains deployment manifests for running the full system on AWS EKS or any Kubernetes cluster:

- **Namespace** for resource isolation
- **OpenSearch** deployment with resource limits
- **API** deployment with 2 replicas, readiness probes, and liveness probes
- **Frontend** deployment with 2 replicas and a LoadBalancer service
- **Ingress** for URL-based routing

The API deployment includes health check probes that Kubernetes uses to automatically restart unhealthy pods and route traffic only to pods that are ready to serve requests.


## Important!

This tool is for informational and educational purposes only. It is not a substitute for professional medical advice. Always consult your doctor before enrolling in a clinical trial. The system searches publicly available data from ClinicalTrials.gov and the FDA.

