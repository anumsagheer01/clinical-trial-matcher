"""
Reindex all trials with vector embeddings.

This script:
1. Loads the parsed trial data
2. Generates a vector embedding for each trial's search text
3. Deletes the old index
4. Creates a new index with vector support
5. Bulk-indexes everything (text + vectors)
"""

import json
import os
import sys

from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.search.index_config_v2 import INDEX_NAME, INDEX_MAPPING
from src.search.embeddings import generate_embeddings_batch

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


def main():
    print("=" * 60)
    print("REINDEXING TRIALS WITH VECTOR EMBEDDINGS")
    print("=" * 60)

    # Load data
    data_path = os.path.join("data", "processed", "trials_cleaned.json")
    print(f"\n  Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    print(f"  Loaded {len(trials)} trials")

    # Generate embeddings for each trial's search_text
    # search_text = title + summary + conditions + keywords + eligibility
    print("\n  Generating vector embeddings (this takes a while)...")
    search_texts = [trial.get("search_text", "")[:512] for trial in trials]
    # Truncate to 512 chars because the model has a max input length
    # and longer text doesn't help much for search relevance

    embeddings = generate_embeddings_batch(search_texts, batch_size=128)
    print(f"  Generated {len(embeddings)} embeddings")

    # Attach embeddings to trials
    for trial, embedding in zip(trials, embeddings):
        trial["search_vector"] = embedding

    # Recreate the index
    client = get_client()

    if client.indices.exists(index=INDEX_NAME):
        print(f"\n  Deleting old index '{INDEX_NAME}'...")
        client.indices.delete(index=INDEX_NAME)

    print(f"  Creating new index with vector support...")
    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

    # Bulk index
    print(f"\n  Indexing {len(trials)} trials with vectors...")

    chunk_size = 500
    success_total = 0
    error_total = 0

    actions = [
        {
            "_index": INDEX_NAME,
            "_id": trial["nct_id"],
            "_source": trial,
        }
        for trial in trials
    ]

    for i in tqdm(range(0, len(actions), chunk_size), desc="  Indexing"):
        chunk = actions[i:i + chunk_size]
        try:
            success, errors = helpers.bulk(
                client, chunk,
                raise_on_error=False,
                raise_on_exception=False,
            )
            success_total += success
            if errors:
                error_total += len(errors)
        except Exception as e:
            print(f"\n  Error: {e}")
            error_total += len(chunk)

    client.indices.refresh(index=INDEX_NAME)

    count = client.count(index=INDEX_NAME)["count"]
    print(f"\n  Done! Indexed: {success_total} | Errors: {error_total}")
    print(f"  Documents in index: {count}")


if __name__ == "__main__":
    main()