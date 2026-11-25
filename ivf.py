"""A minimal IVF (inverted file) index, built with KMeans and numpy. No Qdrant.

Idea: group the posts into clusters once. At search time, only look inside the
few clusters whose centre is closest to the query, instead of all 6000 posts.
"""

from collections import Counter
import json

import numpy as np
from sklearn.cluster import KMeans

from config import IVF_N_CLUSTERS, METADATA_FILE, SEED, VECTORS_NORM_FILE


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
