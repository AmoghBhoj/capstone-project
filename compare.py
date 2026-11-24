"""Searches Qdrant with the 5 test queries and saves the results to results/.

Run `python embed.py` and `python qdrant_setup.py` first.
"""

import json
import time

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
    LATENCY_REPEATS,
    QDRANT_URL,
    QUERIES_RAW_FILE,
    RESULTS_DIR,
    TOP_K,
    WARMUP_RUNS,
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


def median_latency_ms(client, collection, query_vector, search_params):
    """Time one search properly: warm up, repeat, and keep the median (in ms)."""
    # Warm-up: the first searches pay one-time costs, so don't count them.
    for _ in range(WARMUP_RUNS):
        search_ids(client, collection, query_vector, search_params)

    timings = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()                 # precise stopwatch: start
        search_ids(client, collection, query_vector, search_params)
        seconds = time.perf_counter() - start       # stop
        timings.append(seconds * 1000)              # seconds -> milliseconds
    # Median = the middle value, so one slow outlier doesn't distort it.
    return float(np.median(timings))


def run_hnsw(client):
    """Part 3: exact search vs HNSW (default and weak map) at several ef values."""
    query_vectors = np.load(QUERIES_RAW_FILE)

    # Every configuration to test: (name, collection, search settings).
    # "exact" checks all 6000 posts and is our ground truth.
    configs = [("exact", COLLECTION_HNSW_DEFAULT, models.SearchParams(exact=True))]
    for label, collection in HNSW_COLLECTIONS.items():
        for ef in EF_VALUES:
            configs.append((f"{label}_ef{ef}", collection, models.SearchParams(hnsw_ef=ef)))

    # Ground truth first: the exact top-5 ids for each query.
    exact_ids = []
    for query_vector in query_vectors:
        exact_ids.append(search_ids(client, COLLECTION_HNSW_DEFAULT, query_vector,
                                    models.SearchParams(exact=True)))

    results = {}
    for name, collection, search_params in configs:
        ids_per_query = []
        overlaps = []
        latencies = []
        for i, query_vector in enumerate(query_vectors):
            ids = search_ids(client, collection, query_vector, search_params)
            ids_per_query.append(ids)
            overlaps.append(overlap_at_k(ids, exact_ids[i]))
            latencies.append(median_latency_ms(client, collection, query_vector, search_params))
        results[name] = {
            "ids": ids_per_query,
            "overlap_at_5": overlaps,
            "latency_ms": [round(x, 3) for x in latencies],
            "avg_overlap": sum(overlaps) / len(overlaps),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        }

    rows = []
    for name, r in results.items():
        rows.append([name] + r["overlap_at_5"] + [r["avg_overlap"], r["avg_latency_ms"]])
    headers = (["config"] + [f"q{i + 1}" for i in range(len(query_vectors))]
               + ["avg overlap", "avg latency (ms)"])
    print("\nPart 3: overlap@5 with exact search (1.0 = all of the true top 5 found)")
    print(tabulate(rows, headers=headers, floatfmt=".2f"))

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "hnsw.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"Saved {output_file.name}: {len(results)} configurations x {len(query_vectors)} queries")
    return results


def main():
    client = QdrantClient(url=QDRANT_URL)
    run_distance_metrics(client)
    run_hnsw(client)


if __name__ == "__main__":
    main()
