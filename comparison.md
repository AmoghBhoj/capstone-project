# Comparison of Vector Search Methods

## Part 2: Distance metrics (Cosine vs Euclidean vs Dot)

All three collections (`news_cosine`, `news_euclid`, `news_dot`) hold the **same raw
vectors** for the same 6000 posts. Only the distance metric differs. The ranking order
differed between metrics for **all 5** test queries (full results:
[results/distance_metrics.json](results/distance_metrics.json)).

### Example: "any plans for new space exploration by a space agency"

| Rank | Cosine | length | cosine score | Dot | length | cosine score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4756 sci.space | 1.56 | 0.564 | 2172 sci.space | 4.08 | 0.518 |
| 2 | 207 sci.space | 1.57 | 0.555 | 5480 sci.space | 4.10 | 0.472 |
| 3 | 1933 sci.space | 2.06 | 0.543 | **5105 rec.sport.baseball** | **3.80** | **0.391** |
| 4 | 624 sci.space | 1.69 | 0.532 | 5116 sci.space | 2.61 | 0.498 |
| 5 | 2172 sci.space | 4.08 | 0.518 | 4548 sci.space | 3.50 | 0.370 |

*Length = the raw vector's length. The average across all 6000 posts is 2.35.*

**What differs.** Cosine's #1 result is post 4756, but dot's #1 is post 2172. Dot also
ranks post 5105 at #3, which belongs to `rec.sport.baseball`, a topic unrelated to the
query. Cosine doesn't have it in its top 5 at all.

**What each metric measures.**

- **Cosine** measures the **angle only**, so a vector's length doesn't matter.
- **Euclidean** measures the **straight-line distance** between the two vectors' tips,
  so it is affected by both the angle and the difference in length. Lower = closer.
- **Dot product** measures **angle × length**, so it is **magnitude-sensitive**: a longer
  vector gets a higher score.

**Why the order differs.** The baseball post has a low cosine score (0.391), well below
cosine's top-5 cutoff (0.518), so its angle to the query is poor. But its vector length
is 3.80, well above the average of 2.35. Dot multiplies angle by length, so the long
vector makes up for the poor angle and the post ranks #3. All five of dot's results have
above-average lengths, while cosine's top 4 are all below average.

**Why the lengths differ at all.** We uploaded **raw** vectors (the model without its
Normalize layer), so vector lengths vary. Short, focused texts get **longer** vectors,
because their token vectors point the same way. Long, varied texts get **shorter**
vectors, because their token vectors partly cancel out when averaged. The baseball post
is a short one-liner, which is why its vector is long. If we had used normalised vectors
(all length 1), all three metrics would give **exactly the same ranking**: dot would
equal cosine, and Euclidean distance would depend only on the angle.

**Which metric to use in practice.** `all-MiniLM-L6-v2` is trained for cosine similarity,
so cosine (or dot on normalised vectors) is the right choice for this model. Here, the
vector length is a side effect of text length, not of meaning.

### Known limitation: imperfect cleaning

scikit-learn removes headers, signatures and quotes with simple formatting rules, so some
slip through. Post 207 (cosine #2) contains quoted text from another person that wasn't
removed, because those lines start with a space before the `>`. Its most query-relevant
words ("Manned Lunar Exploration") come from that quoted text, so it ranks high partly
because of someone else's words.

## Part 3: HNSW vs exact search

Two Cosine collections hold the same raw vectors and differ only in how the HNSW map
is built. Ground truth is exact (brute-force) search on `news_hnsw_default`, which
checks all 6000 posts. Overlap@5 = how many of the true top 5 a configuration also
found (order ignored). Latency = median of 20 timed searches per query after 2 warm-up
searches, averaged over the 5 queries (full results:
[results/hnsw.json](results/hnsw.json)).

| Config | Map (build settings) | ef | Avg overlap@5 | Avg latency (ms) |
| --- | --- | --- | --- | --- |
| exact | none (checks all 6000) | – | 1.00 | 1.94 |
| default_ef16 | m=16, ef_construct=100 | 16 | 1.00 | 1.64 |
| default_ef64 | m=16, ef_construct=100 | 64 | 1.00 | 1.68 |
| default_ef128 | m=16, ef_construct=100 | 128 | 1.00 | 1.82 |
| weak_ef16 | m=4, ef_construct=8 | 16 | **0.60** | 1.61 |
| weak_ef64 | m=4, ef_construct=8 | 64 | **0.88** | 1.63 |
| weak_ef128 | m=4, ef_construct=8 | 128 | **0.88** | 1.66 |

**Default map, accuracy.** With the default settings (m=16, ef_construct=100), HNSW
found 5 of the true top 5 for every query, even at the smallest ef we tried (16). With
only 6000 posts and a good map, HNSW is as accurate as exact search.

**Weak map, accuracy.** With the weak settings (m=4, ef_construct=8), accuracy dropped
to 0.60 at ef=16, because each post has very few, carelessly chosen neighbours. For
query 1 it found 0 of the true top 5. Raising ef to 64 improved it to 0.88, but ef=128
gave no further benefit: with so few connections, the search runs out of new posts to
reach, so a bigger shortlist has nothing to add.

**What fixes a weak map.** ef is a search-time setting, while m and ef_construct are
build-time settings. A bigger ef helps a bad map only partly (0.60 → 0.88, never 1.00).
The real fix is to rebuild the collection with a higher m (more neighbours per post) and
a higher ef_construct (neighbours chosen more carefully).

**Speed.** HNSW was only about 15% faster than exact search (1.64 vs 1.94 ms). Every
search includes a round trip between Python and Qdrant, which costs about 1.2 ms on its
own (measured with a request that does no search). Without it, HNSW does roughly 40%
less work (about 0.44 vs 0.74 ms), but with only 6000 posts the actual search work is
small, so the round trip hides most of the difference. With millions of posts, exact
search would slow down dramatically while HNSW would stay fast. For the default map, a
bigger ef only costs time (1.64 → 1.82 ms), because accuracy is already 1.00.
Run-to-run differences of a few hundredths of a millisecond are noise.

**Trap we had to avoid: Qdrant skips HNSW for small collections.** Qdrant has two
per-segment rules: `indexing_threshold` decides whether the HNSW map is **built**, and
`full_scan_threshold` decides whether it is **used**. Both default to 10 MB. Our
collections have 2 segments of about 4.5 MB each, below both defaults, so without
lowering both (to 100 KB and 10 KB), every "HNSW" search would really have been an
exact search, showing 100% overlap even for the weak map.

## Part 4: Custom IVF index (KMeans + nprobe)

Our own IVF index (`ivf.py`, numpy and scikit-learn only, no Qdrant) groups the 6000
**normalised** post vectors into 40 clusters with KMeans (seed 42), about 150 posts per
cluster. Ground truth, overlap@5 and latency are measured exactly as in Part 3 (full
results: [results/ivf.json](results/ivf.json)).

| Config | Clusters searched | Avg posts scanned | Avg overlap@5 | Avg latency (ms) |
| --- | --- | --- | --- | --- |
| ivf_nprobe1 | 1 of 40 | 136 | **0.84** | 0.026 |
| ivf_nprobe8 | 8 of 40 | 1271 | 1.00 | 0.118 |
| ivf_nprobe40 | 40 of 40 (= exact) | 6000 | 1.00 | 1.062 |

Per-query overlap at nprobe = 1: q1 1.0, q2 0.8, q3 1.0, q4 **0.4**, q5 1.0.

**What IVF does.** Instead of comparing the query with all 6000 posts, IVF divides the
posts into clusters once, ahead of time. Each cluster has a centroid: the average of all
the vectors in it. When a query comes in, IVF first compares it only with the 40
centroids (Euclidean distance, the same rule KMeans used to build the clusters), keeps
the nprobe closest clusters, and then scores only the posts inside those clusters.

**Accuracy.** With nprobe = 1, IVF searches only one cluster and ignores every post
outside it, even posts that would have scored higher. Because the clusters follow topics
well, this was often enough: 3 of 5 queries still found all of the true top 5 (average
0.84). It fails when a true neighbour sits just across a cluster boundary (queries 2 and
4). With nprobe = 8 (about 20% of the posts), all 5 queries found 5 of 5, but that is not
guaranteed: a true neighbour could sit in the 9th-closest cluster. A larger nprobe can
never do worse than a smaller one, because it scores every post the smaller one scored,
plus more. nprobe = 40 searches every cluster, so it scores every post and is exact
search. It is our sanity check.

**Why nprobe = 1 misses: query 4 (Adam and Eve).** IVF first ranks the clusters by the
Euclidean distance between the query and each centroid. For query 4, the two closest
clusters are almost tied: cluster 20 at 0.958 and cluster 0 at 0.966. nprobe = 1
searched only cluster 20, so it never scored post 3495 (the true #1, score 0.647) or post
3754, both in cluster 0. This happens because the religion newsgroups (christian, atheism,
religion.misc) discuss the same subjects, so their posts are spread over several
neighbouring clusters. Post 715 sits in the 8th-closest cluster, so nprobe = 8 only just
found it.

**Speed.** Qdrant has no built-in IVF index, so we wrote ours with numpy and
scikit-learn. Its latency grows with the number of posts it scores: 136 posts at
nprobe = 1 took 0.026 ms, 1271 posts at nprobe = 8 took 0.118 ms, and all 6000 at
nprobe = 40 took 1.06 ms. So the real speed-up comes from scanning fewer posts: nprobe = 1
was about 40 times faster than a full scan. IVF also looks faster than Qdrant's HNSW
(about 1.6 ms), but that comparison is not fair: our IVF runs inside Python, while every
Qdrant search includes an HTTP round trip of about 1.2 ms. How well IVF works in a real
system also depends on the data: how cleanly it splits into clusters and how many
vectors there are.

**Trade-off.** Raising nprobe brings the results closer to exact search, but each extra
cluster means more posts to score, so it costs more time. At nprobe = 40 we are back to
checking every post, and the speed advantage is gone. The goal is to pick the smallest
nprobe that gives good-enough accuracy. For our 6000 posts and 5 queries, nprobe = 8
found every true top-5 post while scanning only about 21% of the posts.
