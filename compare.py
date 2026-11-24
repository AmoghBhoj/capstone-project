"""Searches Qdrant with the 5 test queries and saves the results to results/.

Run `python embed.py` and `python qdrant_setup.py` first.
"""

import json

import numpy as np
from qdrant_client import QdrantClient

from config import (
    COLLECTION_COSINE,
    COLLECTION_DOT,
    COLLECTION_EUCLID,
    QDRANT_URL,
    QUERIES_RAW_FILE,
    RESULTS_DIR,
    TOP_K,
)
from embed import load_queries

# Part 2: the label used in the results file -> the collection to search.
METRIC_COLLECTIONS = {
    "cosine": COLLECTION_COSINE,
    "euclidean": COLLECTION_EUCLID,
    "dot": COLLECTION_DOT,
}


def search(client, collection, query_vector):
    """Return the TOP_K closest posts in one collection, best first."""
    response = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=TOP_K,
        with_payload=True,   # also send back category + text, not just ids
    )
    results = []
    # enumerate(..., start=1) numbers the results 1, 2, 3... for the rank.
    for rank, point in enumerate(response.points, start=1):
        # Squash newlines and repeated spaces, so the snippet is one tidy line.
        snippet = " ".join(point.payload["text"].split())[:120]
        results.append({
            "rank": rank,
            "id": point.id,
            "score": round(point.score, 4),
            "category": point.payload["category"],
            "snippet": snippet,
        })
    return results


def run_distance_metrics(client):
    """Part 2: run every query against all 3 metric collections and save it."""
    # Query text from queries.md (for labels) and the matching RAW vectors
    # from embed.py. Both are in the same order, so zip() pairs them up.
    queries = load_queries()
    query_vectors = np.load(QUERIES_RAW_FILE)

    all_results = {}
    for query_text, query_vector in zip(queries, query_vectors):
        all_results[query_text] = {}
        for metric, collection in METRIC_COLLECTIONS.items():
            all_results[query_text][metric] = search(client, collection, query_vector)

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "distance_metrics.json"
    # indent=2 makes the file readable for humans, one field per line.
    output_file.write_text(json.dumps(all_results, indent=2))
    print(f"Saved {output_file.name}: {len(all_results)} queries x "
          f"{len(METRIC_COLLECTIONS)} metrics x top-{TOP_K}")
    return all_results


def main():
    client = QdrantClient(url=QDRANT_URL)
    run_distance_metrics(client)


if __name__ == "__main__":
    main()
