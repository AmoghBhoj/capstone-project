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
