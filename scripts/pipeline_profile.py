"""Shared run-profile utilities for the FinanceBench notebook pipeline.

The bootstrap notebook applies the profile exactly once, before PDF parsing.
Every downstream notebook then consumes the resulting working dataset and its
artifacts, so a pilot run cannot silently mix different query subsets.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import pandas as pd


PROFILE_ENVIRONMENT_VARIABLE = "FINANCEBENCH_RUN_PROFILE"
SAMPLE_SIZE_ENVIRONMENT_VARIABLE = "FINANCEBENCH_SAMPLE_SIZE"
SAMPLE_SEED_ENVIRONMENT_VARIABLE = "FINANCEBENCH_SAMPLE_SEED"
PROFILE_MANIFEST_RELATIVE_PATH = Path("data/interim/pipeline_profile.json")

DEFAULT_PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "full": {
        "sample_size": None,
        "sample_seed": 42,
        "publication_eligible": True,
    },
    "pilot": {
        "sample_size": 15,
        "sample_seed": 42,
        "publication_eligible": False,
    },
}


def resolve_profile_settings(
    profile: str,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    definitions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve and validate one full or pilot run profile."""

    available = definitions or DEFAULT_PROFILE_DEFINITIONS
    profile_name = str(profile).strip().lower()
    if profile_name not in available:
        raise ValueError(
            f"Unknown run profile {profile_name!r}; expected one of "
            f"{sorted(available)}"
        )

    definition = dict(available[profile_name])
    effective_size = (
        definition.get("sample_size") if sample_size is None else sample_size
    )
    effective_seed = (
        definition.get("sample_seed", 42) if sample_seed is None else sample_seed
    )
    publication_eligible = bool(definition.get("publication_eligible", False))

    if profile_name == "full" and effective_size is not None:
        raise ValueError("The full profile cannot use --sample-size")
    if profile_name != "full":
        if effective_size is None or int(effective_size) <= 0:
            raise ValueError("A pilot profile requires a positive sample size")
        effective_size = int(effective_size)
    if int(effective_seed) < 0:
        raise ValueError("Sample seed must be non-negative")

    return {
        "name": profile_name,
        "sample_size": effective_size,
        "sample_seed": int(effective_seed),
        "publication_eligible": publication_eligible,
    }


def profile_settings_from_environment() -> dict[str, Any]:
    """Read profile settings inherited by an executed notebook kernel."""

    profile = os.getenv(PROFILE_ENVIRONMENT_VARIABLE, "full")
    raw_size = os.getenv(SAMPLE_SIZE_ENVIRONMENT_VARIABLE)
    raw_seed = os.getenv(SAMPLE_SEED_ENVIRONMENT_VARIABLE)
    sample_size = int(raw_size) if raw_size not in {None, ""} else None
    sample_seed = int(raw_seed) if raw_seed not in {None, ""} else None
    return resolve_profile_settings(
        profile=profile,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )


def export_profile_environment(settings: Mapping[str, Any]) -> None:
    """Export resolved settings for all notebook subprocesses."""

    os.environ[PROFILE_ENVIRONMENT_VARIABLE] = str(settings["name"])
    sample_size = settings.get("sample_size")
    if sample_size is None:
        os.environ.pop(SAMPLE_SIZE_ENVIRONMENT_VARIABLE, None)
    else:
        os.environ[SAMPLE_SIZE_ENVIRONMENT_VARIABLE] = str(int(sample_size))
    os.environ[SAMPLE_SEED_ENVIRONMENT_VARIABLE] = str(int(settings["sample_seed"]))


def _stable_score(seed: int, *parts: object) -> str:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_stratum(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).str.strip().replace("", "unknown")


def _pilot_ids(
    questions: pd.DataFrame,
    documents: pd.DataFrame,
    sample_size: int,
    sample_seed: int,
) -> list[str]:
    """Select a deterministic sample balanced by question and document type."""

    if sample_size > len(questions):
        raise ValueError(
            f"Pilot sample size {sample_size} exceeds the {len(questions)} available questions"
        )

    sampling = questions.copy()
    if "doc_type" not in sampling.columns and "doc_type" in documents.columns:
        sampling = sampling.merge(
            documents[["doc_name", "doc_type"]].drop_duplicates("doc_name"),
            on="doc_name",
            how="left",
        )
    if "question_type" not in sampling.columns:
        sampling["question_type"] = "unknown"
    if "doc_type" not in sampling.columns:
        sampling["doc_type"] = "unknown"

    sampling["_question_stratum"] = _normalise_stratum(sampling["question_type"])
    sampling["_document_stratum"] = _normalise_stratum(sampling["doc_type"])
    sampling["_score"] = sampling["financebench_id"].map(
        lambda value: _stable_score(sample_seed, value)
    )

    question_strata = sorted(sampling["_question_stratum"].unique())
    base_quota, remainder = divmod(sample_size, len(question_strata))
    quotas = {
        stratum: base_quota + (index < remainder)
        for index, stratum in enumerate(question_strata)
    }

    selected: list[str] = []
    selected_set: set[str] = set()
    for question_stratum in question_strata:
        group = sampling[sampling["_question_stratum"] == question_stratum]
        quota = min(quotas[question_stratum], len(group))
        document_strata = sorted(
            group["_document_stratum"].unique(),
            key=lambda value: _stable_score(sample_seed, question_stratum, value),
        )

        # First cover as many document types as the stratum quota permits.
        for document_stratum in document_strata:
            if quota <= 0:
                break
            candidates = group[
                (group["_document_stratum"] == document_stratum)
                & (~group["financebench_id"].astype(str).isin(selected_set))
            ].sort_values(["_score", "financebench_id"])
            if candidates.empty:
                continue
            financebench_id = str(candidates.iloc[0]["financebench_id"])
            selected.append(financebench_id)
            selected_set.add(financebench_id)
            quota -= 1

        if quota > 0:
            remaining = group[
                ~group["financebench_id"].astype(str).isin(selected_set)
            ].sort_values(["_score", "financebench_id"])
            for financebench_id in remaining["financebench_id"].astype(str).head(quota):
                selected.append(financebench_id)
                selected_set.add(financebench_id)

    # Small strata may not satisfy their initial quota. Fill globally and
    # preserve the exact requested sample size.
    if len(selected) < sample_size:
        remaining = sampling[
            ~sampling["financebench_id"].astype(str).isin(selected_set)
        ].sort_values(["_score", "financebench_id"])
        for financebench_id in remaining["financebench_id"].astype(str).head(
            sample_size - len(selected)
        ):
            selected.append(financebench_id)
            selected_set.add(financebench_id)

    if len(selected) != sample_size or len(selected_set) != sample_size:
        raise RuntimeError("Pilot sampling did not produce the requested unique IDs")
    return selected


def apply_profile_to_dataset(
    questions: pd.DataFrame,
    documents: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply a profile before PDF download/parsing and build its manifest."""

    required_question_columns = {"financebench_id", "doc_name"}
    missing = sorted(required_question_columns - set(questions.columns))
    if missing:
        raise ValueError(f"FinanceBench questions are missing columns: {missing}")
    if questions["financebench_id"].astype(str).duplicated().any():
        raise ValueError("financebench_id must be unique before profile selection")

    source_question_count = len(questions)
    if settings["name"] == "full":
        selected_ids = questions["financebench_id"].astype(str).tolist()
    else:
        selected_ids = _pilot_ids(
            questions=questions,
            documents=documents,
            sample_size=int(settings["sample_size"]),
            sample_seed=int(settings["sample_seed"]),
        )

    selection_order = {value: index for index, value in enumerate(selected_ids)}
    selected_questions = questions[
        questions["financebench_id"].astype(str).isin(selection_order)
    ].copy()
    selected_questions["_profile_order"] = (
        selected_questions["financebench_id"].astype(str).map(selection_order)
    )
    selected_questions = (
        selected_questions.sort_values("_profile_order")
        .drop(columns="_profile_order")
        .reset_index(drop=True)
    )

    selected_doc_names = sorted(
        selected_questions["doc_name"].dropna().astype(str).unique()
    )
    selected_documents = documents[
        documents["doc_name"].astype(str).isin(selected_doc_names)
    ].copy().reset_index(drop=True)
    missing_documents = sorted(
        set(selected_doc_names) - set(selected_documents["doc_name"].astype(str))
    )
    if missing_documents:
        raise ValueError(
            "Profile selection has no document metadata for: "
            + ", ".join(missing_documents[:10])
        )

    summary_frame = selected_questions.copy()
    if (
        "doc_type" not in summary_frame.columns
        and "doc_type" in selected_documents.columns
    ):
        summary_frame = summary_frame.merge(
            selected_documents[["doc_name", "doc_type"]].drop_duplicates("doc_name"),
            on="doc_name",
            how="left",
        )

    selection_sha256 = hashlib.sha256(
        ("\n".join(selected_ids) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": settings["name"],
        "publication_eligible": bool(settings["publication_eligible"]),
        "sampling_strategy": (
            "all_questions"
            if settings["name"] == "full"
            else "deterministic_question_type_document_type_stratified"
        ),
        "sample_seed": int(settings["sample_seed"]),
        "requested_sample_size": settings.get("sample_size"),
        "source_question_count": int(source_question_count),
        "actual_question_count": int(len(selected_questions)),
        "actual_document_count": int(len(selected_doc_names)),
        "selected_financebench_ids": selected_ids,
        "selected_doc_names": selected_doc_names,
        "selection_sha256": selection_sha256,
        "question_type_counts": (
            summary_frame["question_type"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
            if "question_type" in summary_frame.columns
            else {}
        ),
        "document_type_counts": (
            summary_frame["doc_type"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
            if "doc_type" in summary_frame.columns
            else {}
        ),
    }
    return selected_questions, selected_documents, manifest


def write_profile_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
