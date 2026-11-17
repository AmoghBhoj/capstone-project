# Qdrant Capstone: Comparing Vector Search Algorithms

Embeds the 20 Newsgroups dataset with `all-MiniLM-L6-v2`, loads it into a local
Qdrant instance, and compares distance metrics, HNSW settings, and a hand-written
IVF index.

## Requirements

- Python **3.12+** (the pinned numpy version needs 3.12 or newer)
- Docker with Docker Compose (Docker Desktop, or Colima on macOS)

## Part 0: Setup

### 1. Create the virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start Qdrant

If you use Colima instead of Docker Desktop, start its VM first with `colima start`.

```bash
docker compose up -d
```

This runs `qdrant/qdrant:v1.20.2` in the background with:

- port **6333**: REST API and web dashboard
- port **6334**: gRPC API
- data stored in `./qdrant_storage/` (git-ignored), so collections survive restarts

### 3. Check that it is running

```bash
curl http://localhost:6333/readyz        # prints: all shards are ready
```

Then open the dashboard: <http://localhost:6333/dashboard>.
It should load and show **no collections** yet. They are created in Parts 2 and 3.

To stop Qdrant: `docker compose down` (your data in `qdrant_storage/` is kept).

<!-- TODO: add a screenshot of the empty dashboard here, e.g. docs/dashboard.png -->
