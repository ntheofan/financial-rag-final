"""Reusable helpers for metadata-aware, hybrid-preserving reranking."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_rerank_candidate_pool(
    candidates: pd.DataFrame,
    *,
    top_n: int,
) -> None:
    required = {
        "financebench_id",
        "retrieved_rank",
        "chunk_id",
        "retrieved_doc_id",
        "chunk_text",
        "rrf_score",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Missing reranking candidate columns: {missing}")

    counts = candidates.groupby("financebench_id").size()
    insufficient = counts[counts < top_n]
    if len(insufficient):
        raise RuntimeError(
            f"{len(insufficient)} queries have fewer than top_n={top_n} candidates"
        )

    duplicate_mask = candidates.duplicated(
        subset=["financebench_id", "chunk_id"], keep=False
    )
    if duplicate_mask.any():
        duplicate_queries = candidates.loc[
            duplicate_mask, "financebench_id"
        ].nunique()
        raise RuntimeError(
            f"Duplicate chunk candidates detected for {duplicate_queries} queries"
        )


def build_reranker_document(candidate: Any) -> str:
    """Prefix chunk text with non-gold metadata available at retrieval time."""
    doc_id = str(candidate.get("retrieved_doc_id", "")).strip()
    chunk_id = str(candidate.get("chunk_id", "")).strip()
    chunk_text = str(candidate.get("chunk_text", "")).strip()
    return (
        f"Document ID: {doc_id}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Content:\n{chunk_text}"
    )


def _minmax(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    minimum = numeric.min()
    maximum = numeric.max()
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Scores contain no finite values")
    if np.isclose(maximum, minimum):
        return pd.Series(np.ones(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)


def fuse_rerank_scores(
    candidates: pd.DataFrame,
    *,
    rerank_weight: float = 0.7,
    hybrid_weight: float = 0.3,
) -> pd.DataFrame:
    if not np.isclose(rerank_weight + hybrid_weight, 1.0):
        raise ValueError("rerank_weight and hybrid_weight must sum to 1.0")
    if rerank_weight < 0 or hybrid_weight < 0:
        raise ValueError("score fusion weights must be non-negative")

    frame = candidates.copy()
    frame["rerank_score_normalized"] = _minmax(frame["rerank_score"])
    frame["rrf_score_normalized"] = _minmax(frame["rrf_score"])
    frame["fused_score"] = (
        rerank_weight * frame["rerank_score_normalized"]
        + hybrid_weight * frame["rrf_score_normalized"]
    )

    pure_order = frame.sort_values(
        ["rerank_score", "retrieved_rank"],
        ascending=[False, True],
        kind="stable",
    )
    pure_rank = pd.Series(
        range(1, len(pure_order) + 1), index=pure_order.index, dtype=int
    )
    frame["pure_rerank_rank"] = pure_rank.reindex(frame.index).astype(int)

    frame = frame.sort_values(
        ["fused_score", "rerank_score", "retrieved_rank"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["fused_rank"] = range(1, len(frame) + 1)
    return frame
