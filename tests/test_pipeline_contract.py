import json
import re
import unittest
from pathlib import Path

from scripts.run_kaggle_pipeline import (
    REPO_ROOT,
    load_config,
    required_outputs_for_stage,
    selected_notebooks,
    validate_notebooks,
)


CONFIG_PATH = REPO_ROOT / "configs" / "kaggle_pipeline.json"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        cls.notebook_names = selected_notebooks(cls.config, "all")

    def test_pipeline_contains_exactly_the_numbered_notebooks(self) -> None:
        self.assertEqual(len(self.notebook_names), 15)
        expected_prefixes = [f"{number:02d}_" for number in range(1, 16)]
        actual_prefixes = [name[:3] for name in self.notebook_names]
        self.assertEqual(actual_prefixes, expected_prefixes)
        validate_notebooks(self.notebook_names)

    def test_notebooks_are_valid_json_with_unique_cell_ids(self) -> None:
        for notebook_name in self.notebook_names:
            with self.subTest(notebook=notebook_name):
                payload = json.loads(
                    (NOTEBOOKS_DIR / notebook_name).read_text(encoding="utf-8")
                )
                self.assertEqual(payload.get("nbformat"), 4)
                cell_ids = [
                    cell.get("id")
                    for cell in payload.get("cells", [])
                    if cell.get("id")
                ]
                self.assertEqual(len(cell_ids), len(set(cell_ids)))

                for cell in payload.get("cells", []):
                    if cell.get("cell_type") != "code":
                        continue
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs", []), [])

    def test_required_outputs_are_unique_relative_paths(self) -> None:
        outputs = required_outputs_for_stage(self.config, "all")
        self.assertEqual(len(outputs), len(set(outputs)))
        for output in outputs:
            path = Path(output)
            self.assertFalse(path.is_absolute(), output)
            self.assertNotIn("..", path.parts, output)

    def test_canonical_notebooks_reject_placeholder_results(self) -> None:
        notebook_sources = {
            name: (NOTEBOOKS_DIR / name).read_text(encoding="utf-8")
            for name in (
                "10_answer_generation.ipynb",
                "13_ragas_evaluation.ipynb",
                "14_ragchecker_evaluation.ipynb",
            )
        }
        self.assertIn(
            "FALLBACK_TO_DRY_RUN_WITHOUT_OPENAI_KEY = False",
            notebook_sources["10_answer_generation.ipynb"],
        )
        self.assertIn(
            "cell_validate_ragas_results",
            notebook_sources["13_ragas_evaluation.ipynb"],
        )
        self.assertIn(
            "validate_ragchecker_results",
            notebook_sources["14_ragchecker_evaluation.ipynb"],
        )

    def test_reranking_candidate_contract_is_explicit(self) -> None:
        contracts = self.config["contracts"]
        self.assertEqual(contracts["hybrid_candidates_per_query"], 20)
        self.assertEqual(contracts["reranked_results_per_query"], 10)

        hybrid_source = (
            NOTEBOOKS_DIR / "08_hybrid_retrieval.ipynb"
        ).read_text(encoding="utf-8")
        reranker_source = (
            NOTEBOOKS_DIR / "09_reranking.ipynb"
        ).read_text(encoding="utf-8")
        self.assertIn("TOP_K = 20", hybrid_source)
        self.assertIn("RERANK_TOP_N = 20", reranker_source)
        self.assertIn("FINAL_TOP_K = 10", reranker_source)
        self.assertIn("validate_rerank_candidate_pool", reranker_source)

    def test_kaggle_dataset_contract_is_explicit(self) -> None:
        dataset = self.config["kaggle_dataset"]
        self.assertEqual(
            dataset["handle"],
            "theofanisnikolaou/financebench-sample-dataset",
        )
        self.assertEqual(dataset["mount_slug"], "financebench-sample-dataset")

        bootstrap_source = (
            NOTEBOOKS_DIR / "02_prepare_financebench_dataset.ipynb"
        ).read_text(encoding="utf-8")
        self.assertIn("bootstrap_attached_kaggle_dataset", bootstrap_source)
        self.assertIn("FINANCEBENCH_KAGGLE_DATASET_DIR", bootstrap_source)

    def test_notebooks_have_no_hardcoded_personal_kaggle_slug(self) -> None:
        for notebook_name in self.notebook_names:
            source = (NOTEBOOKS_DIR / notebook_name).read_text(encoding="utf-8")
            with self.subTest(notebook=notebook_name):
                self.assertNotIn("/kaggle/input/datasets/theofanisnikolaou", source)

        notebook_03 = (
            NOTEBOOKS_DIR / "03_parse_pdfs_with_docling.ipynb"
        ).read_text(encoding="utf-8")
        self.assertIn("pdf_search_dirs = [PDFS_DIR]", notebook_03)

    def test_no_obvious_api_keys_in_tracked_source_candidates(self) -> None:
        secret_patterns = (
            re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
        )
        candidates = [
            *NOTEBOOKS_DIR.glob("*.ipynb"),
            *REPO_ROOT.glob("*.py"),
            *REPO_ROOT.glob("*.md"),
            *REPO_ROOT.glob("*.txt"),
            *REPO_ROOT.glob("*.json"),
        ]
        for path in candidates:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in secret_patterns:
                with self.subTest(path=path, pattern=pattern.pattern):
                    self.assertIsNone(pattern.search(text))


if __name__ == "__main__":
    unittest.main()
