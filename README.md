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

## Part 1: Dataset and embeddings

Qdrant is not needed for this part. It only uses Python and files.

```bash
python embed.py
```

This one command:

1. Loads the 20 Newsgroups posts (downloaded automatically on the first run),
   removes headers, signatures and quoted replies, drops posts under 50 characters,
   and picks 6000 at random with seed 42 (`data.py`).
2. Turns each post and each query in `queries.md` into a 384-number vector with
   `all-MiniLM-L6-v2` (the model is downloaded automatically on the first run).
3. Saves everything to `artifacts/` (git-ignored):

| File | Contents |
| --- | --- |
| `vectors_raw.npy` | 6000 post vectors, raw (lengths vary) |
| `vectors_norm.npy` | 6000 post vectors, normalised (length 1) |
| `metadata.json` | text and topic of each post (`id` = row number in the vector files) |
| `queries_raw.npy` / `queries_norm.npy` | the 5 query vectors, raw and normalised |

Running it again skips the posts (about 30 seconds of work) and only redoes the
5 queries. To redo everything: `python embed.py --force`.

To preview the sample without embedding: `python data.py`.

## Part 2: Distance metrics

Needs Qdrant running (Part 0) and the files from Part 1.

```bash
python qdrant_setup.py   # creates the collections and uploads the 6000 posts (~5 s)
python compare.py        # searches with the 5 queries and saves the results
```

`qdrant_setup.py` creates three collections holding the **same raw vectors** and payload
(category + text). Only the distance metric differs:

| Collection | Distance | Measures |
| --- | --- | --- |
| `news_cosine` | Cosine | angle only (higher = closer) |
| `news_euclid` | Euclidean | straight-line distance (lower = closer) |
| `news_dot` | Dot product | angle × length (higher = closer) |

Running it again deletes and recreates these three collections, so it always starts clean.

`compare.py` runs each query against all three collections, prints the top 5 side by
side, flags queries where the **ranking order** differs, and saves everything to
`results/distance_metrics.json`. The findings are written up in `comparison.md`.

If you edit `queries.md`, run `python embed.py` again before `compare.py`, so the
query vectors match the new text.
