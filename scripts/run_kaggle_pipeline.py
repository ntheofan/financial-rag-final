"""Execute the canonical Financial RAG notebook pipeline on Kaggle.

The runner deliberately uses a clean-process boundary for every notebook. Data
is passed only through declared artifacts under ``data/``. A successful run
produces an immutable run directory with logs, executed notebooks, a manifest,
SHA-256 checksums, and an optional ZIP bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "kaggle_pipeline.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(directory: Path) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
        total_bytes += size
    return {
        "path": str(directory),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def resolve_kaggle_dataset(
    config: dict[str, Any], require_dataset: bool
) -> dict[str, Any] | None:
    dataset_config = config.get("kaggle_dataset")
    if not dataset_config:
        if require_dataset:
            raise RuntimeError("No kaggle_dataset is configured")
        return None

    environment_variable = dataset_config.get(
        "environment_variable", "FINANCEBENCH_KAGGLE_DATASET_DIR"
    )
    explicit_path = os.getenv(environment_variable)
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    kaggle_input = Path("/kaggle/input")
    mount_slug = dataset_config.get("mount_slug")
    if mount_slug:
        candidates.append(kaggle_input / mount_slug)

    dataset_path = next((path for path in candidates if path.is_dir()), None)
    in_kaggle = Path("/kaggle/working").is_dir() or bool(
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
    )

    if dataset_path is None and in_kaggle:
        try:
            import kagglehub

            downloaded_path = kagglehub.dataset_download(dataset_config["handle"])
            dataset_path = Path(downloaded_path)
        except Exception as exc:
            if require_dataset:
                raise RuntimeError(
                    f"Unable to resolve Kaggle dataset {dataset_config['handle']}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    if dataset_path is None:
        if require_dataset:
            raise FileNotFoundError(
                "FinanceBench Kaggle dataset was not mounted or downloadable. "
                f"Attach {dataset_config['handle']} as a Kaggle Input."
            )
        return None

    dataset_path = dataset_path.resolve()
    os.environ[environment_variable] = str(dataset_path)
    fingerprint = dataset_fingerprint(dataset_path)
    fingerprint["handle"] = dataset_config["handle"]
    fingerprint["environment_variable"] = environment_variable
    return fingerprint


def capture(command: list[str], cwd: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    commit = capture(["git", "rev-parse", "HEAD"])
    branch = capture(["git", "branch", "--show-current"])
    status = capture(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"stage_order", "stages", "api_notebooks", "required_outputs"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Pipeline config is missing keys: {missing}")

    configured = []
    for stage in config["stage_order"]:
        if stage not in config["stages"]:
            raise ValueError(f"Stage {stage!r} is absent from stages")
        configured.extend(config["stages"][stage])

    duplicates = sorted({name for name in configured if configured.count(name) > 1})
    if duplicates:
        raise ValueError(f"Notebooks occur in multiple stages: {duplicates}")
    return config


def selected_notebooks(config: dict[str, Any], stage: str) -> list[str]:
    if stage == "all":
        return [
            notebook
            for stage_name in config["stage_order"]
            for notebook in config["stages"][stage_name]
        ]
    return list(config["stages"][stage])


def validate_notebooks(notebook_names: Iterable[str]) -> None:
    missing = [
        str(REPO_ROOT / "notebooks" / name)
        for name in notebook_names
        if not (REPO_ROOT / "notebooks" / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing notebooks:\n- " + "\n- ".join(missing))


def existing_data_artifacts() -> list[Path]:
    artifacts = []
    for relative in ("data/interim", "data/processed"):
        directory = REPO_ROOT / relative
        if not directory.exists():
            continue
        artifacts.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        )
    return sorted(artifacts)


def stream_command(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def execute_notebook(
    notebook_name: str,
    executed_dir: Path,
    logs_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    notebook_path = REPO_ROOT / "notebooks" / notebook_name
    executed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        notebook_name,
        "--output-dir",
        str(executed_dir),
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]

    started = time.monotonic()
    started_at = utc_now()
    return_code = stream_command(
        command,
        cwd=REPO_ROOT,
        log_path=logs_dir / f"{Path(notebook_name).stem}.log",
    )
    duration = time.monotonic() - started
    return {
        "notebook": notebook_name,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": round(duration, 3),
        "return_code": return_code,
        "status": "completed" if return_code == 0 else "failed",
        "source_sha256": sha256_file(notebook_path),
    }


def required_outputs_for_stage(
    config: dict[str, Any], stage: str
) -> list[str]:
    if stage == "all":
        return [
            output
            for stage_name in config["stage_order"]
            for output in config["required_outputs"].get(stage_name, [])
        ]
    return list(config["required_outputs"].get(stage, []))


def validate_required_outputs(relative_paths: Iterable[str]) -> None:
    failures = []
    for relative in relative_paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif path.stat().st_size == 0:
            failures.append(f"empty: {relative}")
    if failures:
        raise RuntimeError("Required output validation failed:\n- " + "\n- ".join(failures))


def validate_api_outputs(notebook_names: Iterable[str]) -> None:
    selected = set(notebook_names)

    if "10_answer_generation.ipynb" in selected:
        for run_name in ("dense", "hybrid", "hybrid_reranked"):
            path = REPO_ROOT / "data" / "processed" / "qa_results" / f"rag_qa_stats_{run_name}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("generation_mode") != "openai":
                raise RuntimeError(f"{path.name} was not generated in openai mode")
            if int(payload.get("failed_generations", 0)) != 0:
                raise RuntimeError(f"{path.name} contains failed generations")

    if "13_ragas_evaluation.ipynb" in selected:
        import pandas as pd

        path = REPO_ROOT / "data" / "processed" / "evaluation" / "ragas_results.csv"
        frame = pd.read_csv(path)
        if "ragas_status" not in frame.columns:
            raise RuntimeError("ragas_results.csv has no ragas_status column")
        statuses = set(frame["ragas_status"].dropna().astype(str))
        if statuses != {"completed"}:
            raise RuntimeError(f"RAGAS did not complete successfully: {sorted(statuses)}")

    if "14_ragchecker_evaluation.ipynb" in selected:
        for run_name in ("dense", "hybrid", "reranked"):
            path = REPO_ROOT / "data" / "processed" / "evaluation" / f"ragchecker_results_{run_name}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("ragchecker_status") != "completed":
                raise RuntimeError(f"RAGChecker did not complete successfully for {run_name}")


def validate_retrieval_outputs(
    config: dict[str, Any], notebook_names: Iterable[str]
) -> None:
    selected = set(notebook_names)
    contracts = config.get("contracts", {})

    if "08_hybrid_retrieval.ipynb" in selected:
        import pandas as pd

        expected = int(contracts.get("hybrid_candidates_per_query", 20))
        path = (
            REPO_ROOT
            / "data"
            / "processed"
            / "retrieval_results"
            / "retrieval_results_hybrid.csv"
        )
        frame = pd.read_csv(path, usecols=["financebench_id", "retrieved_rank"])
        counts = frame.groupby("financebench_id").size()
        invalid = counts[counts != expected]
        if len(invalid):
            raise RuntimeError(
                f"Hybrid candidate contract failed for {len(invalid)} queries; "
                f"expected exactly {expected} candidates per query"
            )

    if "09_reranking.ipynb" in selected:
        import pandas as pd

        expected = int(contracts.get("reranked_results_per_query", 10))
        results_path = (
            REPO_ROOT
            / "data"
            / "processed"
            / "retrieval_results"
            / "retrieval_results_hybrid_reranked.csv"
        )
        manifest_path = (
            REPO_ROOT
            / "data"
            / "processed"
            / "retrieval_results"
            / "retrieval_manifest_hybrid_reranked.csv"
        )
        frame = pd.read_csv(results_path, usecols=["financebench_id", "retrieved_rank"])
        counts = frame.groupby("financebench_id").size()
        invalid = counts[counts != expected]
        if len(invalid):
            raise RuntimeError(
                f"Reranked output contract failed for {len(invalid)} queries; "
                f"expected exactly {expected} results per query"
            )

        manifest = pd.read_csv(manifest_path)
        failed = manifest[manifest["status"] != "success"]
        if len(failed):
            raise RuntimeError(f"Reranking failed for {len(failed)} queries")

        expected_input = int(contracts.get("hybrid_candidates_per_query", 20))
        undersized = manifest[manifest["n_input_candidates"] < expected_input]
        if len(undersized):
            raise RuntimeError(
                f"Reranker received fewer than {expected_input} candidates for "
                f"{len(undersized)} queries"
            )


def validate_stage_checkpoint(config: dict[str, Any], stage: str) -> None:
    stage_notebooks = config["stages"][stage]
    validate_required_outputs(config["required_outputs"].get(stage, []))
    validate_retrieval_outputs(config, stage_notebooks)
    validate_api_outputs(stage_notebooks)


def copy_bundle_artifacts(config: dict[str, Any], artifacts_dir: Path) -> list[str]:
    copied = []
    for relative in config.get("bundle_paths", []):
        source = REPO_ROOT / relative
        if not source.exists():
            continue
        destination = artifacts_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        copied.append(relative)
    return copied


def write_checksums(run_dir: Path) -> Path:
    checksum_path = run_dir / "checksums.sha256"
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(run_dir).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Pipeline JSON configuration.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        help="all or one configured stage.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable run identifier; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Directory that receives run metadata and bundles.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-notebook execution timeout in seconds; -1 disables it.",
    )
    parser.add_argument(
        "--allow-existing-artifacts",
        action="store_true",
        help="Allow a full run to start with existing data artifacts.",
    )
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Allow a full run from a Git worktree with uncommitted changes.",
    )
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        help="Bypass the OPENAI_API_KEY preflight check.",
    )
    parser.add_argument(
        "--require-kaggle-dataset",
        action="store_true",
        help="Fail unless the configured Kaggle dataset can be resolved.",
    )
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Do not create the final ZIP artifact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the selected notebooks without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    valid_stages = {"all", *config["stage_order"]}
    if args.stage not in valid_stages:
        raise ValueError(f"Unknown stage {args.stage!r}; expected one of {sorted(valid_stages)}")

    notebooks = selected_notebooks(config, args.stage)
    validate_notebooks(notebooks)

    if args.require_kaggle_dataset:
        os.environ["FINANCEBENCH_REQUIRE_KAGGLE_DATASET"] = "1"
    dataset = resolve_kaggle_dataset(
        config,
        require_dataset=args.require_kaggle_dataset,
    )

    current_git_state = git_state()
    if (
        args.stage == "all"
        and not args.dry_run
        and current_git_state.get("dirty")
        and not args.allow_dirty_worktree
    ):
        raise RuntimeError(
            "A canonical full run requires a clean Git worktree. Commit the "
            "intended source revision or pass --allow-dirty-worktree explicitly."
        )

    api_selected = sorted(set(notebooks) & set(config["api_notebooks"]))
    if api_selected and not os.getenv("OPENAI_API_KEY") and not args.allow_missing_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required by: " + ", ".join(api_selected)
        )

    if args.stage == "all" and not args.allow_existing_artifacts:
        existing = existing_data_artifacts()
        if existing:
            preview = "\n- ".join(str(path.relative_to(REPO_ROOT)) for path in existing[:20])
            raise RuntimeError(
                "A canonical full run must start without existing interim/processed artifacts. "
                "Use a clean clone or pass --allow-existing-artifacts explicitly.\n- " + preview
            )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.run_root.resolve()
    run_dir = run_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    executed_dir = run_dir / "executed_notebooks"
    logs_dir = run_dir / "logs"
    manifest_path = run_dir / "run_manifest.json"
    run_dir.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "dry_run" if args.dry_run else "running",
        "stage": args.stage,
        "started_at": utc_now(),
        "finished_at": None,
        "repo_root": str(REPO_ROOT),
        "git": current_git_state,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "kaggle_kernel_run_type": os.getenv("KAGGLE_KERNEL_RUN_TYPE"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        },
        "input_dataset": dataset,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "notebooks": notebooks,
        "notebook_runs": [],
        "required_outputs": required_outputs_for_stage(config, args.stage),
        "copied_bundle_paths": [],
        "bundle_zip": None,
        "error": None,
    }
    atomic_write_json(manifest_path, manifest)

    environment = capture([sys.executable, "-m", "pip", "freeze"])
    (run_dir / "environment.txt").write_text(
        (environment or "pip freeze unavailable") + "\n",
        encoding="utf-8",
    )

    print(f"Run ID: {run_id}")
    print(f"Stage: {args.stage}")
    print("Notebooks:")
    for notebook in notebooks:
        print(f"- {notebook}")

    if args.dry_run:
        manifest["finished_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        write_checksums(run_dir)
        return 0

    try:
        for notebook in notebooks:
            print(f"\n{'=' * 80}\nRunning {notebook}\n{'=' * 80}")
            result = execute_notebook(
                notebook,
                executed_dir=executed_dir,
                logs_dir=logs_dir,
                timeout=args.timeout,
            )
            manifest["notebook_runs"].append(result)
            atomic_write_json(manifest_path, manifest)
            if result["return_code"] != 0:
                raise RuntimeError(f"Notebook failed: {notebook}")

            completed_stage = next(
                (
                    stage_name
                    for stage_name in config["stage_order"]
                    if notebook == config["stages"][stage_name][-1]
                ),
                None,
            )
            if completed_stage is not None:
                print(f"Validating completed stage: {completed_stage}")
                validate_stage_checkpoint(config, completed_stage)

        validate_required_outputs(manifest["required_outputs"])
        validate_retrieval_outputs(config, notebooks)
        validate_api_outputs(notebooks)

        artifacts_dir = run_dir / "artifacts"
        manifest["copied_bundle_paths"] = copy_bundle_artifacts(config, artifacts_dir)
        manifest["status"] = "completed"
        manifest["finished_at"] = utc_now()
        if not args.no_bundle:
            manifest["bundle_zip"] = str(run_root / f"{run_id}.zip")
        atomic_write_json(manifest_path, manifest)
        write_checksums(run_dir)

        if not args.no_bundle:
            archive_base = run_root / run_id
            bundle_path = Path(
                shutil.make_archive(
                    str(archive_base),
                    "zip",
                    root_dir=run_root,
                    base_dir=run_id,
                )
            )
            print(f"Bundle: {bundle_path}")

        print(f"Completed run {run_id}")
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(manifest_path, manifest)
        write_checksums(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
