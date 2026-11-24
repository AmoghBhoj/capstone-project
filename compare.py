"""Searches Qdrant with the 5 test queries and saves the results to results/.

Run `python embed.py` and `python qdrant_setup.py` first.
"""

import json

import numpy as np
from qdrant_client import QdrantClient, models
from tabulate import tabulate

from config import (
    COLLECTION_COSINE,
    COLLECTION_DOT,
    COLLECTION_EUCLID,
    COLLECTION_HNSW_DEFAULT,
    COLLECTION_HNSW_WEAK,
    EF_VALUES,
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

# Part 3: label -> HNSW collection (both Cosine, different map quality).
HNSW_COLLECTIONS = {
    "default": COLLECTION_HNSW_DEFAULT,
    "weak": COLLECTION_HNSW_WEAK,
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


def order_differs(results_per_metric):
    """True if the metrics did NOT all return the same ids in the same order.

    This is "rule B": the brief asks for queries where the ranking ORDER
    differs, so swapping two posts already counts as a difference.
    """
    first_ids = None
    for metric in METRIC_COLLECTIONS:
        ids = [hit["id"] for hit in results_per_metric[metric]]   # e.g. [4756, 207, ...]
        if first_ids is None:
            first_ids = ids          # remember the first metric's order
        elif ids != first_ids:       # lists are equal only if same ids in same order
            return True
    return False


def print_side_by_side(query_text, results_per_metric):
    """Print one table per query: a row per rank, a column per metric."""
    rows = []
    for i in range(TOP_K):
        row = [i + 1]   # rank 1..TOP_K
        for metric in METRIC_COLLECTIONS:
            hit = results_per_metric[metric][i]
            row.append(f"{hit['id']:<5} {hit['category']:<24} {hit['score']}")
        rows.append(row)

    flag = "ORDER DIFFERS" if order_differs(results_per_metric) else "same order"
    print(f'\nQuery: "{query_text}"   -> {flag}')
    print(tabulate(rows, headers=["rank"] + list(METRIC_COLLECTIONS)))


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
        print_side_by_side(query_text, all_results[query_text])

    # Summary: which queries are candidates for the comparison.md write-up.
    differing = []
    for query_text in all_results:
        if order_differs(all_results[query_text]):
            differing.append(query_text)
    print(f"\nRanking order differs in {len(differing)} of {len(all_results)} queries.")

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "distance_metrics.json"
    # indent=2 makes the file readable for humans, one field per line.
    output_file.write_text(json.dumps(all_results, indent=2))
    print(f"Saved {output_file.name}: {len(all_results)} queries x "
          f"{len(METRIC_COLLECTIONS)} metrics x top-{TOP_K}")
    return all_results


def search_ids(client, collection, query_vector, search_params):
    """Return only the ids of the TOP_K results, best first."""
    response = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=TOP_K,
        search_params=search_params,   # exact search, or HNSW with a given ef
    )
    ids = []
    for point in response.points:
        ids.append(point.id)
    return ids


def overlap_at_k(found_ids, true_ids):
    """Share of the true top-K that was also found: 5 of 5 -> 1.0, 3 of 5 -> 0.6.

    Uses sets, so only WHICH posts were found matters, not their order.
    """
    both = set(found_ids) & set(true_ids)   # & = the ids that are in both
    return len(both) / TOP_K


def run_hnsw(client):
    """Part 3: how often does HNSW find the same top 5 as exact search?"""
    query_vectors = np.load(QUERIES_RAW_FILE)

    # Ground truth: exact (brute-force) search, which checks all 6000 posts.
    exact_ids = []
    for query_vector in query_vectors:
        ids = search_ids(client, COLLECTION_HNSW_DEFAULT, query_vector,
                         models.SearchParams(exact=True))
        exact_ids.append(ids)

    # Every collection x every ef: search all queries, compare with the truth.
    rows = []
    for label, collection in HNSW_COLLECTIONS.items():
        for ef in EF_VALUES:
            overlaps = []
            for i, query_vector in enumerate(query_vectors):
                ids = search_ids(client, collection, query_vector,
                                 models.SearchParams(hnsw_ef=ef))
                overlaps.append(overlap_at_k(ids, exact_ids[i]))
            average = sum(overlaps) / len(overlaps)
            rows.append([f"{label}_ef{ef}"] + overlaps + [average])

    headers = ["config"] + [f"q{i + 1}" for i in range(len(query_vectors))] + ["avg"]
    print("\nPart 3: overlap@5 with exact search (1.0 = found all of the true top 5)")
    print(tabulate(rows, headers=headers, floatfmt=".2f"))


def main():
    client = QdrantClient(url=QDRANT_URL)
    run_distance_metrics(client)
    run_hnsw(client)


if __name__ == "__main__":
    main()
