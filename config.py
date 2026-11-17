"""Shared constants used by every script, so they only live in one place."""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Build paths from this file's location, so scripts work no matter which
# folder you run them from.
PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"   # embeddings + metadata (git-ignored)
RESULTS_DIR = PROJECT_DIR / "results"       # JSON/CSV results (committed)
QUERIES_FILE = PROJECT_DIR / "queries.md"

# --- Qdrant --------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"   # matches the port in docker-compose.yml
