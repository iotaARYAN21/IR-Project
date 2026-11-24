"""Maximal Marginal Relevance (MMR): re-order results to reduce redundancy."""

import re
from collections.abc import Sequence

# Very common words say nothing about whether two passages repeat each other.
STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her his in is it "
    "its of on or she that the their there they this to was were which who will "
    "with".split()
)


def word_set(text: str) -> frozenset[str]:
    """Lower-cased words of a text, without very common words."""
    return frozenset(re.findall(r"\w+", text.lower())) - STOPWORDS


def jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    """Shared words divided by all words in either text (0 to 1)."""
    union = len(first | second)
    return len(first & second) / union if union else 0.0


def passage_similarity(
    titles: Sequence[str], texts: Sequence[str]
) -> list[list[float]]:
    """Similarity of every pair of passages, from 0 (unrelated) to 1 (repeat).

    A pair scores 1 when both passages have the same title (same article).
    Otherwise it scores the Jaccard similarity of their words, so passages from
    different articles that repeat each other still count.
    """
    if len(titles) != len(texts):
        raise ValueError("titles and texts must have the same length")
    clean_titles = [title.strip().lower() for title in titles]
    words = [word_set(f"{title} {text}") for title, text in zip(titles, texts)]
    count = len(titles)
    matrix = [[1.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            same_article = bool(clean_titles[i]) and clean_titles[i] == clean_titles[j]
            value = 1.0 if same_article else jaccard(words[i], words[j])
            matrix[i][j] = matrix[j][i] = value
    return matrix


def normalize(scores: Sequence[float]) -> list[float]:
    """Scale scores to 0..1 (min-max), so BM25, cosine and CrossEncoder scores mix
    on one scale. If every score is equal they all become 1.0."""
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def mmr_order(
    relevance: Sequence[float],
    similarity: Sequence[Sequence[float]],
    k: int,
    diversity: float,
) -> list[int]:
    """Indices of the k candidates chosen by MMR, in the order they were picked.

    `relevance[i]` is the score of candidate i (higher is better) and
    `similarity[i][j]` is how similar candidates i and j are (1 = identical).
    """
    count = len(relevance)
    if not 0 <= diversity <= 1:
        raise ValueError("diversity must be between 0 and 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(similarity) != count or any(len(row) != count for row in similarity):
        raise ValueError("similarity must be a square matrix matching relevance")
    if count == 0:
        return []

    scaled = normalize(relevance)
    remaining = list(range(count))
    chosen: list[int] = []
    # Highest similarity from each candidate to anything chosen so far.
    closest = [0.0] * count
    while remaining and len(chosen) < k:
        best = max(
            remaining,
            key=lambda i: (
                (1 - diversity) * scaled[i] - diversity * closest[i],
                -i,  # a tie keeps the original (better ranked) candidate
            ),
        )
        chosen.append(best)
        remaining.remove(best)
        for i in remaining:
            closest[i] = max(closest[i], similarity[i][best])
    return chosen
