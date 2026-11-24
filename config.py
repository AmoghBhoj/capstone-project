"""Shared constants used by every script, so they only live in one place."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Build paths from this file's location, so scripts work no matter which
# folder you run them from.
PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"   # embeddings + metadata (git-ignored)
RESULTS_DIR = PROJECT_DIR / "results"       # JSON/CSV results (committed)
QUERIES_FILE = PROJECT_DIR / "queries.md"

# Files written by embed.py. Row i of each posts file = post i in
# metadata.json (and later, point id i in Qdrant).
VECTORS_RAW_FILE = ARTIFACTS_DIR / "vectors_raw.npy"     # posts, raw (lengths vary)
VECTORS_NORM_FILE = ARTIFACTS_DIR / "vectors_norm.npy"   # posts, normalised (length 1)
METADATA_FILE = ARTIFACTS_DIR / "metadata.json"          # text + category of each post
QUERIES_RAW_FILE = ARTIFACTS_DIR / "queries_raw.npy"     # 5 queries, raw
QUERIES_NORM_FILE = ARTIFACTS_DIR / "queries_norm.npy"   # 5 queries, normalised

# --- Embedding model -----------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384      # how many numbers (dimensions) each vector has

# --- Dataset -------------------------------------------------------------
SEED = 42              # fixed seed, so the same 6000 posts are picked every run
SAMPLE_SIZE = 6000     # how many posts we keep (the brief allows 5,000-8,000)
MIN_DOC_CHARS = 50     # posts shorter than this after cleaning are dropped

# --- Qdrant --------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"   # matches the port in docker-compose.yml

# Part 2: same raw vectors, three distance methods
COLLECTION_COSINE = "news_cosine"
COLLECTION_EUCLID = "news_euclid"
COLLECTION_DOT = "news_dot"
