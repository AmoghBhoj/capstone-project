"""Creates the Qdrant collections and uploads the post vectors into them.

Run `python embed.py` first, so the vectors exist in artifacts/.
"""

import json

import numpy as np
from qdrant_client import QdrantClient, models

from config import (
    COLLECTION_COSINE,
    COLLECTION_DOT,
    COLLECTION_EUCLID,
    METADATA_FILE,
    QDRANT_URL,
    VECTOR_SIZE,
    VECTORS_RAW_FILE,
)

# Part 2: one collection per distance method. The method is fixed when a
# collection is created, so comparing 3 methods needs 3 collections.
METRIC_COLLECTIONS = {
    COLLECTION_COSINE: models.Distance.COSINE,   # angle only
    COLLECTION_EUCLID: models.Distance.EUCLID,   # straight-line distance
    COLLECTION_DOT: models.Distance.DOT,         # angle x lengths
}


def create_collection(client, name, distance):
    """Create an empty collection, deleting any old one with the same name."""
    # Delete and recreate, so running the script again starts clean.
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=distance),
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


if __name__ == "__main__":
    main()
