"""Creates the Qdrant collections and uploads the post vectors into them.

Run `python embed.py` first, so the vectors exist in artifacts/.
"""

import json
import time

import numpy as np
from qdrant_client import QdrantClient, models

from config import (
    COLLECTION_COSINE,
    COLLECTION_DOT,
    COLLECTION_EUCLID,
    COLLECTION_HNSW_DEFAULT,
    COLLECTION_HNSW_WEAK,
    FULL_SCAN_THRESHOLD_KB,
    INDEXING_THRESHOLD_KB,
    METADATA_FILE,
    QDRANT_URL,
    VECTOR_SIZE,
    VECTORS_RAW_FILE,
    WEAK_EF_CONSTRUCT,
    WEAK_M,
)

# Part 2: one collection per distance method. The method is fixed when a
# collection is created, so comparing 3 methods needs 3 collections.
METRIC_COLLECTIONS = {
    COLLECTION_COSINE: models.Distance.COSINE,   # angle only
    COLLECTION_EUCLID: models.Distance.EUCLID,   # straight-line distance
    COLLECTION_DOT: models.Distance.DOT,         # angle x lengths
}

# Part 3: two Cosine collections that differ only in how well the HNSW map
# is built. full_scan_threshold is lowered in both, otherwise Qdrant would
# skip the map for a collection this small and do an exact search instead.
HNSW_COLLECTIONS = {
    # Default m (16) and ef_construct (100): a well-built map.
    COLLECTION_HNSW_DEFAULT: models.HnswConfigDiff(
        full_scan_threshold=FULL_SCAN_THRESHOLD_KB,
    ),
    # Few friends per post, chosen carelessly: a deliberately poor map.
    COLLECTION_HNSW_WEAK: models.HnswConfigDiff(
        m=WEAK_M,
        ef_construct=WEAK_EF_CONSTRUCT,
        full_scan_threshold=FULL_SCAN_THRESHOLD_KB,
    ),
}


def create_collection(client, name, distance, hnsw_config=None, optimizers_config=None):
    """Create an empty collection, deleting any old one with the same name.

    hnsw_config and optimizers_config are only used by the Part 3 collections;
    None means "use Qdrant's defaults".
    """
    # Delete and recreate, so running the script again starts clean.
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=distance),
        hnsw_config=hnsw_config,
        optimizers_config=optimizers_config,
    )


def upload_posts(client, name, vectors, metadata):
    """Upload every post as a point: id + vector + payload."""
    points = []
    for i in range(len(vectors)):
        points.append(models.PointStruct(
            id=i,                          # row i of the .npy file = post i
            vector=vectors[i].tolist(),    # numpy row -> plain Python list
            payload={
                "category": metadata[i]["category"],
                "text": metadata[i]["text"],
            },
        ))
    # Sends the points in batches of 256 instead of all 6000 at once.
    # wait=True: don't return until Qdrant has actually stored them.
    client.upload_points(collection_name=name, points=points,
                         batch_size=256, wait=True)


def wait_for_index(client, name, timeout_seconds=120):
    """Wait until Qdrant has finished building the HNSW map for this collection.

    Qdrant builds the map in the background after the upload. "green" means
    it's idle, and indexed_vectors_count tells us how many posts are in the map.
    """
    for _ in range(timeout_seconds):
        info = client.get_collection(name)
        indexed = info.indexed_vectors_count or 0
        if info.status == models.CollectionStatus.GREEN and indexed >= info.points_count:
            break
        time.sleep(1)   # not ready yet, check again in a second
    print(f"{name:<18} status={info.status.value}  "
          f"indexed={indexed}/{info.points_count}")


def main():
    client = QdrantClient(url=QDRANT_URL)

    # RAW vectors (lengths vary), so the 3 methods can disagree.
    vectors = np.load(VECTORS_RAW_FILE)
    metadata = json.loads(METADATA_FILE.read_text())

    for name, distance in METRIC_COLLECTIONS.items():
        create_collection(client, name, distance)
        upload_posts(client, name, vectors, metadata)
        count = client.count(collection_name=name).count
        print(f"{name:<12} distance={distance.value:<7} points={count}")

    # Part 3: same raw vectors. Cosine normalises them internally anyway.
    print()
    for name, hnsw_config in HNSW_COLLECTIONS.items():
        create_collection(
            client, name, models.Distance.COSINE,
            hnsw_config=hnsw_config,
            # Build the map even though the collection is small.
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=INDEXING_THRESHOLD_KB,
            ),
        )
        upload_posts(client, name, vectors, metadata)
    # Wait after uploading both, so the two maps are built at the same time.
    for name in HNSW_COLLECTIONS:
        wait_for_index(client, name)


if __name__ == "__main__":
    main()
