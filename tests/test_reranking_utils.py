import unittest

import pandas as pd

from scripts.reranking_utils import (
    build_reranker_document,
    fuse_rerank_scores,
    validate_rerank_candidate_pool,
)


class MetadataAwareRerankingTests(unittest.TestCase):
    def test_reranker_document_exposes_document_identity(self) -> None:
        text = build_reranker_document(
            {
                "retrieved_doc_id": "LOCKHEEDMARTIN_2021_10K",
                "chunk_id": "chunk_0342",
                "chunk_text": "Total current assets were 19,815.",
                "expected_doc_name": "SHOULD_NOT_BE_USED",
            }
        )
        self.assertIn("LOCKHEEDMARTIN_2021_10K", text)
        self.assertIn("Total current assets", text)
        self.assertNotIn("SHOULD_NOT_BE_USED", text)

    def test_fusion_retains_hybrid_signal(self) -> None:
        candidates = pd.DataFrame(
            {
                "retrieved_rank": [1, 2, 3],
                "rrf_score": [0.10, 0.09, 0.01],
                "rerank_score": [0.7000, 0.6900, 0.7001],
            }
        )
        fused = fuse_rerank_scores(
            candidates,
            rerank_weight=0.7,
            hybrid_weight=0.3,
        )
        self.assertEqual(fused.iloc[0]["retrieved_rank"], 1)
        self.assertEqual(fused["fused_rank"].tolist(), [1, 2, 3])
        self.assertEqual(set(fused["pure_rerank_rank"]), {1, 2, 3})

    def test_candidate_validation_rejects_duplicate_chunks(self) -> None:
        candidates = pd.DataFrame(
            {
                "financebench_id": ["q1", "q1"],
                "retrieved_rank": [1, 2],
                "chunk_id": ["same", "same"],
                "retrieved_doc_id": ["doc", "doc"],
                "chunk_text": ["a", "a"],
                "rrf_score": [0.2, 0.1],
            }
        )
        with self.assertRaises(RuntimeError):
            validate_rerank_candidate_pool(candidates, top_n=2)


if __name__ == "__main__":
    unittest.main()
