"""Loads the 20 Newsgroups posts, cleans them, and picks a random sample."""

import random

from sklearn.datasets import fetch_20newsgroups

from config import MIN_DOC_CHARS, SAMPLE_SIZE, SEED


def load_documents():
    """Return SAMPLE_SIZE posts, each as {"text": ..., "category": ...}."""
    # Get all ~18,800 posts (downloaded the first time, then read from the
    # saved copy). remove=... cuts out headers, signatures and quoted
    # replies, so only the writer's own message is left.
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    documents = []
    for text, topic_number in zip(dataset.data, dataset.target):
        text = text.strip()
        # Some posts are (almost) empty after cleaning, e.g. a reply that
        # was only a quote. Their vectors would mean nothing, so skip them.
        if len(text) < MIN_DOC_CHARS:
            continue
        # Turn the topic number (e.g. 1) into its name (e.g. "comp.graphics").
        category = dataset.target_names[topic_number]
        documents.append({"text": text, "category": category})

    # Pick SAMPLE_SIZE posts at random. The fixed SEED makes it pick the
    # same posts every time, so results can be repeated.
    rng = random.Random(SEED)
    sample = rng.sample(documents, SAMPLE_SIZE)

    print(f"{len(dataset.data)} posts downloaded, "
          f"{len(documents)} left after dropping short ones, "
          f"{len(sample)} picked.")
    return sample


if __name__ == "__main__":
    # Only runs for `python data.py`, to check the result by eye.
    docs = load_documents()
    print("\nFirst picked post, topic:", docs[0]["category"])
    print(docs[0]["text"][:300])
