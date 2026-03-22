"""
Index all parsed clinical trials into OpenSearch.

This script:
1. Connects to OpenSearch
2. Creates the index with our mapping (or recreates it if it exists)
3. Bulk-indexes all trials
4. Verifies the count

BULK INDEXING explained:
Instead of sending 15,000 individual requests (slow), send them
in batches of 500 (fast). OpenSearch processes each batch atomically.
"""

import json
import os
import sys

from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.search.index_config import INDEX_NAME, INDEX_MAPPING

# Load environment variables
load_dotenv()


def get_opensearch_client():
    """
    Create and return an OpenSearch client.
    
    Reads connection details from environment variables (.env file).
    """
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )
    
    # Test the connection
    try:
        info = client.info()
        print(f"  Connected to OpenSearch {info['version']['number']}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to OpenSearch: {e}")
        print("  Make sure OpenSearch is running: docker compose up -d")
        sys.exit(1)
    
    return client


def create_index(client):
    """
    Create the clinical trials index.

    """
    # Delete if exists (clean slate)
    if client.indices.exists(index=INDEX_NAME):
        print(f"  Index '{INDEX_NAME}' already exists. Deleting...")
        client.indices.delete(index=INDEX_NAME)
    
    # Create with our mapping
    print(f"  Creating index '{INDEX_NAME}'...")
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    print(f"  Index created successfully!")


def bulk_index_trials(client, trials):
    """
    Index all trials using the bulk API.
    
    The bulk API sends many documents in one request.
    This is MUCH faster than indexing one at a time.
    
    How it works:
    - Create "actions" - each action says "index this document"
    - Send them in batches of 500
    - OpenSearch processes each batch
    - Track successes and failures
    """
    print(f"\n  Indexing {len(trials)} trials into OpenSearch...")
    
    def generate_actions():
        """Generate bulk actions for each trial."""
        for trial in trials:
            yield {
                "_index": INDEX_NAME,
                "_id": trial["nct_id"],  # Use NCT ID as document ID (prevents duplicates)
                "_source": trial,
            }
    
    # Helper
    success_count = 0
    error_count = 0
    
    # Process in chunks with progress bar
    actions = list(generate_actions())
    chunk_size = 500
    
    for i in tqdm(range(0, len(actions), chunk_size), desc="  Indexing"):
        chunk = actions[i:i + chunk_size]
        try:
            success, errors = helpers.bulk(
                client,
                chunk,
                raise_on_error=False,
                raise_on_exception=False,
            )
            success_count += success
            if errors:
                error_count += len(errors)
                if error_count <= 3:  # Print first few errors for debugging
                    for err in errors[:2]:
                        print(f"\n  ERROR: {err}")
        except Exception as e:
            print(f"\n  Bulk indexing error: {e}")
            error_count += len(chunk)
    
    # Force OpenSearch to make the data searchable immediately
    client.indices.refresh(index=INDEX_NAME)
    
    print(f"\n  Indexing complete!")
    print(f"  Successfully indexed: {success_count}")
    print(f"  Errors: {error_count}")
    
    # Verify count
    count = client.count(index=INDEX_NAME)["count"]
    print(f"  Documents in index: {count}")


def main():
    print("=" * 60)
    print("INDEXING CLINICAL TRIALS INTO OPENSEARCH")
    print("=" * 60)
    
    # Load parsed data
    data_path = os.path.join("data", "processed", "trials_cleaned.json")
    print(f"\n  Loading data from {data_path}...")
    
    with open(data_path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    print(f"  Loaded {len(trials)} trials")
    
    # Connect and index
    client = get_opensearch_client()
    create_index(client)
    bulk_index_trials(client, trials)
    
    print("\n" + "=" * 60)
    print("DONE! Trials are now searchable in OpenSearch.")
    print("=" * 60)


if __name__ == "__main__":
    main()