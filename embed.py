"""Turns the 6000 posts and the 5 queries into vectors and saves them to artifacts/.

Usage:
    python embed.py           # skips the posts if their files already exist
    python embed.py --force   # redoes the posts anyway
"""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    ARTIFACTS_DIR,
    METADATA_FILE,
    MODEL_NAME,
    QUERIES_FILE,
    QUERIES_NORM_FILE,
    QUERIES_RAW_FILE,
    VECTORS_NORM_FILE,
    VECTORS_RAW_FILE,
)
from data import load_documents


def load_queries():
    """Read the numbered lines (like "1. some question") from queries.md."""
    queries = []
    for line in QUERIES_FILE.read_text().splitlines():
        line = line.strip()
        # Keep only lines that start with a number, e.g. "3. some question".
        # Headings, bullet points and empty lines are skipped.
        if line[:1].isdigit() and ". " in line:
            question = line.split(". ", 1)[1]   # the part after "3. "
            queries.append(question)
    return queries


def build_raw_model():
    """Load the model WITHOUT its last station (Normalize).

    The full model has 3 stations: [0] Transformer -> [1] Pooling -> [2] Normalize.
    Normalize sets every vector's length to 1, which would make cosine, dot and
    Euclidean always agree in Part 2. Keeping only stations 0 and 1 gives us
    the raw vectors, whose lengths differ.
    """
    full_model = SentenceTransformer(MODEL_NAME)
    return SentenceTransformer(modules=[full_model[0], full_model[1]])


def normalise(vectors):
    """Divide every vector by its own length, so each one has length 1."""
    # axis=1: one length per row (per vector).
    # keepdims=True: keeps the lengths as a column, so each row is divided
    # by its OWN length.
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / lengths


def print_length_stats(name, vectors):
    """Print the smallest, average and largest vector length."""
    lengths = np.linalg.norm(vectors, axis=1)
    print(f"{name} vector lengths: min={lengths.min():.3f}  "
          f"mean={lengths.mean():.3f}  max={lengths.max():.3f}")


def main():
    # Read the optional --force flag from the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="redo the posts even if their files already exist")
    args = parser.parse_args()

    # artifacts/ is not in git, so on a fresh checkout it doesn't exist yet.
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model = build_raw_model()

    # --- Posts: slow (~30 s), so skip if already done ---------------------
    files_exist = (VECTORS_RAW_FILE.exists()
                   and VECTORS_NORM_FILE.exists()
                   and METADATA_FILE.exists())

    if files_exist and not args.force:
        print("Post vectors already exist, skipping (use --force to redo).")
        vectors_raw = np.load(VECTORS_RAW_FILE)
    else:
        documents = load_documents()                     # from data.py
        texts = [doc["text"] for doc in documents]       # just the text of each post
        vectors_raw = model.encode(texts, batch_size=64, show_progress_bar=True)
        vectors_norm = normalise(vectors_raw)

        np.save(VECTORS_RAW_FILE, vectors_raw)
        np.save(VECTORS_NORM_FILE, vectors_norm)

        # Save each post's text and topic. The id is the row number, so
        # metadata[i] describes row i of both vector files.
        metadata = []
        for i, doc in enumerate(documents):
            metadata.append({"id": i, "category": doc["category"], "text": doc["text"]})
        METADATA_FILE.write_text(json.dumps(metadata))

        print(f"Saved {vectors_raw.shape[0]} post vectors, "
              f"each with {vectors_raw.shape[1]} numbers.")

    print_length_stats("Post", vectors_raw)

    # --- Queries: only 5, so always redo them (in case queries.md changed) -
    # Same raw model as the posts, so posts and queries can be compared.
    queries = load_queries()
    queries_raw = model.encode(queries)
    np.save(QUERIES_RAW_FILE, queries_raw)
    np.save(QUERIES_NORM_FILE, normalise(queries_raw))

    print(f"Saved {len(queries)} query vectors.")
    print_length_stats("Query", queries_raw)


if __name__ == "__main__":
    main()
