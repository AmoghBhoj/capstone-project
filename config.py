"""Shared constants used by every script, so they only live in one place."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Build paths from this file's location, so scripts work no matter which
# folder you run them from.
PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"   # embeddings + metadata (git-ignored)
RESULTS_DIR = PROJECT_DIR / "results"       # JSON/CSV results (committed)
QUERIES_FILE = PROJECT_DIR / "queries.md"

# --- Dataset -------------------------------------------------------------
SEED = 42              # fixed seed, so the same 6000 posts are picked every run
SAMPLE_SIZE = 6000     # how many posts we keep (the brief allows 5,000-8,000)
MIN_DOC_CHARS = 50     # posts shorter than this after cleaning are dropped

# --- Qdrant --------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"   # matches the port in docker-compose.yml
