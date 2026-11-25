"""A minimal IVF (inverted file) index, built with KMeans and numpy. No Qdrant.

Idea: group the posts into clusters once. At search time, only look inside the
few clusters whose centre is closest to the query, instead of all 6000 posts.
"""

from collections import Counter
import json

import numpy as np
from sklearn.cluster import KMeans

from config import (
    IVF_N_CLUSTERS,
    METADATA_FILE,
    QUERIES_NORM_FILE,
    SEED,
    TOP_K,
    VECTORS_NORM_FILE,
)


def build_ivf_index(vectors, n_clusters=IVF_N_CLUSTERS, seed=SEED):
    """Group the vectors into clusters, and remember which posts are in each.

    Use NORMALISED vectors, so posts are grouped by direction (meaning),
    not by vector length.
    """
    # KMeans repeats "assign each post to its nearest centre, move each
    # centre to the average of its posts" until the centres stop moving.
    # random_state fixes the random starting centres, so we get the same
    # clusters on every run.
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(vectors)   # labels[i] = cluster number of post i

    # For each cluster, the ids (row numbers) of the posts that joined it.
    clusters = {}
    for cluster_id in range(n_clusters):
        # np.where(labels == cluster_id)[0] -> positions where the label matches
        clusters[cluster_id] = np.where(labels == cluster_id)[0]

    return {
        "centroids": kmeans.cluster_centers_,   # 40 x 384: the centre of each cluster
        "clusters": clusters,                   # cluster id -> array of post ids
        "vectors": vectors,                     # needed at search time
    }


def search_ivf(index, query_vector, nprobe, k=TOP_K):
    """Search only the nprobe clusters whose centre is closest to the query.

    query_vector must be NORMALISED, like the posts in the index.
    Returns (ids, scores, n_scanned): the top-k post ids, best first, their
    cosine scores, and how many posts were actually scored.
    """
    # Step 1: straight-line (Euclidean) distance from the query to each of the
    # 40 centres. Same rule KMeans used to put each post in its cluster.
    distances = np.linalg.norm(index["centroids"] - query_vector, axis=1)

    # Step 2: the nprobe closest centres. argsort lists positions from the
    # smallest distance to the largest, so the first nprobe are the closest.
    nearest_clusters = np.argsort(distances)[:nprobe]

    # Step 3: the ids of every post in those clusters, joined into one array.
    id_groups = []
    for cluster_id in nearest_clusters:
        id_groups.append(index["clusters"][cluster_id])
    candidate_ids = np.concatenate(id_groups)

    # Step 4: score ONLY those posts. Both sides have length 1, so the dot
    # product equals the cosine score (the same scoring as exact search).
    candidate_vectors = index["vectors"][candidate_ids]   # just those rows
    scores = np.dot(candidate_vectors, query_vector)      # one score per post

    # argsort sorts smallest first; the minus sign flips it to highest first.
    best = np.argsort(-scores)[:k]
    top_ids = candidate_ids[best]
    top_scores = scores[best]

    # .tolist() turns numpy numbers into plain Python ones (needed for JSON later).
    return top_ids.tolist(), top_scores.tolist(), len(candidate_ids)


if __name__ == "__main__":
    # Run `python ivf.py` to check that the clusters look sensible.
    vectors = np.load(VECTORS_NORM_FILE)
    metadata = json.loads(METADATA_FILE.read_text())
    index = build_ivf_index(vectors)

    sizes = [len(ids) for ids in index["clusters"].values()]
    print(f"{len(sizes)} clusters, {sum(sizes)} posts in total. "
          f"Cluster sizes: smallest {min(sizes)}, average {sum(sizes) / len(sizes):.0f}, "
          f"largest {max(sizes)}")

    # Do clusters group posts by topic? Show the most common topic in a few.
    print("\nMost common topic in the first 8 clusters:")
    for cluster_id in range(8):
        ids = index["clusters"][cluster_id]
        topics = Counter(metadata[i]["category"] for i in ids)
        topic, count = topics.most_common(1)[0]
        print(f"  cluster {cluster_id:<2} {len(ids):>4} posts  "
              f"top topic: {topic:<26} ({count / len(ids):.0%} of the cluster)")

    # Try one search: query 2 (the RAM question) at three nprobe values.
    # nprobe = all clusters means every post is scored, i.e. exact search.
    query_vectors = np.load(QUERIES_NORM_FILE)
    query_vector = query_vectors[1]   # row 1 = query 2 (rows start at 0)
    print("\nQuery 2: how far can I expand the RAM in my PC")
    for nprobe in [1, 8, IVF_N_CLUSTERS]:
        ids, scores, n_scanned = search_ivf(index, query_vector, nprobe)
        rounded = [round(s, 3) for s in scores]
        print(f"  nprobe={nprobe:<2}  scanned {n_scanned:>4} posts  ids {ids}  scores {rounded}")
