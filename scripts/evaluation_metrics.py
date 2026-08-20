"""Deterministic, finance-aware metrics shared by evaluation notebooks.

The numeric evaluator keeps textual overlap as a secondary diagnostic and
evaluates financial quantities separately.  It understands common magnitude
suffixes, percentages, accounting negatives, and the unit requested by the
question.  This avoids treating answers such as ``$5,818M`` and ``$5818.00``
as unrelated strings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


_NUMBER_PATTERN = re.compile(
    r"(?P<accounting>\(\s*)?"
    r"(?P<sign>[+\-−–]?)\s*"
    r"(?P<currency>[$€£]?)\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<close_before>\))?"
    r"\s*(?P<suffix>%|percent(?:age)?|bps?|x|times?|k|m|mm|bn|b|"
    r"thousand(?:s)?|million(?:s)?|billion(?:s)?)?"
    r"\s*(?P<close_after>\))?",
    flags=re.IGNORECASE,
)

_UNIT_MULTIPLIERS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
}


@dataclass(frozen=True)
class FinancialNumber:
    raw: str
    value: float
    decimal_places: int
    multiplier: float
    is_percent: bool
    is_basis_points: bool

    @property
    def canonical_value(self) -> float:
        if self.is_basis_points:
            return self.value / 10_000.0
        if self.is_percent:
            return self.value / 100.0
        return self.value * self.multiplier

    @property
    def canonical_rounding_tolerance(self) -> float:
        displayed_tolerance = 0.5 * (10.0 ** (-self.decimal_places))
        if self.is_basis_points:
            return displayed_tolerance / 10_000.0
        if self.is_percent:
            return displayed_tolerance / 100.0
        return displayed_tolerance * self.multiplier


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def normalize_text(text: Any) -> str:
    if _is_missing(text):
        return ""
    normalized = str(text).strip().lower()
    normalized = normalized.replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", normalized)


def normalize_finance_answer(text: Any) -> str:
    normalized = normalize_text(text)
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"(?<=\d)\s*(?:m|mm)\b", " million", normalized)
    normalized = re.sub(r"(?<=\d)\s*(?:b|bn)\b", " billion", normalized)
    normalized = normalized.replace("$", " ").replace("€", " ").replace("£", " ")
    normalized = re.sub(r"\busd\b", " ", normalized)
    normalized = re.sub(r"\bmillions?\b", " ", normalized)
    normalized = re.sub(r"\bbillions?\b", " ", normalized)
    normalized = re.sub(r"\bthousands?\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9\.\-%]+", normalize_finance_answer(text))


def lexical_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(
        min(count, ref_counts.get(token, 0))
        for token, count in pred_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_substring_match(prediction: Any, reference: Any) -> int:
    pred_normalized = normalize_finance_answer(prediction)
    ref_normalized = normalize_finance_answer(reference)
    if not pred_normalized or not ref_normalized:
        return 0
    return int(
        pred_normalized in ref_normalized or ref_normalized in pred_normalized
    )


def context_support_score(answer: Any, context: Any) -> float:
    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(context))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def is_insufficient_evidence(text: Any) -> bool:
    if _is_missing(text):
        return True
    return "insufficient evidence" in str(text).lower()


def infer_question_multiplier(question: Any) -> float:
    normalized = normalize_text(question)
    if re.search(r"\b(?:usd|dollars?)?\s*(?:in\s+)?billions?\b", normalized):
        return 1_000_000_000.0
    if re.search(r"\b(?:usd|dollars?)?\s*(?:in\s+)?millions?\b", normalized):
        return 1_000_000.0
    if re.search(r"\b(?:usd|dollars?)?\s*(?:in\s+)?thousands?\b", normalized):
        return 1_000.0
    return 1.0


def extract_financial_numbers(
    text: Any,
    question: Any = "",
    *,
    conservative: bool = False,
) -> list[FinancialNumber]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    default_multiplier = infer_question_multiplier(question)
    extracted: list[FinancialNumber] = []

    for match in _NUMBER_PATTERN.finditer(normalized):
        number_text = match.group("number")
        suffix = (match.group("suffix") or "").lower()
        if suffix == "x" and not re.search(
            r"\b(?:ratio|times|turnover|multiple)\b", normalize_text(question)
        ):
            suffix = ""
        sign = match.group("sign")
        accounting_negative = bool(
            match.group("accounting")
            and (match.group("close_before") or match.group("close_after"))
        )
        value = float(number_text.replace(",", ""))
        if accounting_negative or sign in {"-", "−", "–"}:
            value = -value

        is_percent = suffix in {"%", "percent", "percentage"}
        is_basis_points = suffix in {"bp", "bps"}
        multiplier = _UNIT_MULTIPLIERS.get(suffix, default_multiplier)

        # Calendar years are context, not answer quantities, unless they carry
        # an explicit financial unit or percentage marker.
        if (
            1900 <= abs(value) <= 2100
            and not suffix
            and not match.group("currency")
        ):
            continue

        decimals = len(number_text.split(".", 1)[1]) if "." in number_text else 0
        if (
            conservative
            and decimals == 0
            and not suffix
            and not match.group("currency")
            and default_multiplier == 1.0
        ):
            # Bare integers are frequently identifiers or counts (for example
            # Boeing 737/777/787). Keep the numeric metric conservative and let
            # the textual metric evaluate those answers.
            continue
        extracted.append(
            FinancialNumber(
                raw=match.group(0).strip(),
                value=value,
                decimal_places=decimals,
                multiplier=multiplier,
                is_percent=is_percent,
                is_basis_points=is_basis_points,
            )
        )

    return extracted


def _candidate_values(number: FinancialNumber) -> list[tuple[float, float]]:
    canonical = (
        number.canonical_value,
        max(number.canonical_rounding_tolerance, 1e-12),
    )
    if number.is_percent or number.is_basis_points:
        # Some FinanceBench gold answers express ratios as decimals while model
        # answers express them as percentages (or vice versa).  Retain both
        # representations and select the closest valid comparison.
        displayed_tolerance = max(
            0.5 * (10.0 ** (-number.decimal_places)), 1e-12
        )
        return [canonical, (number.value, displayed_tolerance)]
    return [canonical]


def _numbers_match(
    prediction: FinancialNumber,
    reference: FinancialNumber,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[bool, float, float]:
    best_absolute_error = math.inf
    best_relative_error = math.inf

    for pred_value, _ in _candidate_values(prediction):
        for ref_value, ref_rounding_tolerance in _candidate_values(reference):
            absolute_error = abs(pred_value - ref_value)
            denominator = max(abs(ref_value), 1e-12)
            relative_error = absolute_error / denominator
            best_absolute_error = min(best_absolute_error, absolute_error)
            best_relative_error = min(best_relative_error, relative_error)
            allowed_absolute = max(absolute_tolerance, ref_rounding_tolerance)
            if absolute_error <= allowed_absolute or relative_error <= relative_tolerance:
                return True, absolute_error, relative_error

    return False, best_absolute_error, best_relative_error


def financial_numeric_metrics(
    prediction: Any,
    reference: Any,
    *,
    question: Any = "",
    relative_tolerance: float = 0.01,
    absolute_tolerance: float = 0.005,
) -> dict[str, Any]:
    pred_numbers = extract_financial_numbers(prediction, question=question)
    ref_numbers = extract_financial_numbers(
        reference,
        question=question,
        conservative=True,
    )

    if not ref_numbers:
        return {
            "numeric_applicable": False,
            "numeric_match": math.nan,
            "numeric_coverage": math.nan,
            "numeric_matched_count": 0,
            "gold_numeric_count": 0,
            "pred_numeric_count": len(pred_numbers),
            "numeric_best_absolute_error": math.nan,
            "numeric_best_relative_error": math.nan,
            "gold_numbers": [],
            "pred_numbers": [number.raw for number in pred_numbers],
        }

    # FinanceBench metric-generated questions usually have one gold quantity,
    # while generated answers may expose several intermediate operands.  In
    # that case score the final reported number, not any coincidental operand.
    scoring_pred_numbers = (
        pred_numbers[-1:] if len(ref_numbers) == 1 and pred_numbers else pred_numbers
    )

    matched_reference_indices: set[int] = set()
    absolute_errors: list[float] = []
    relative_errors: list[float] = []

    for ref_index, ref_number in enumerate(ref_numbers):
        best_match: tuple[float, float] | None = None
        for pred_number in scoring_pred_numbers:
            matched, absolute_error, relative_error = _numbers_match(
                pred_number,
                ref_number,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            if matched and (
                best_match is None or relative_error < best_match[1]
            ):
                best_match = (absolute_error, relative_error)
        if best_match is not None:
            matched_reference_indices.add(ref_index)
            absolute_errors.append(best_match[0])
            relative_errors.append(best_match[1])

    matched_count = len(matched_reference_indices)
    coverage = matched_count / len(ref_numbers)
    return {
        "numeric_applicable": True,
        "numeric_match": float(matched_count == len(ref_numbers)),
        "numeric_coverage": float(coverage),
        "numeric_matched_count": matched_count,
        "gold_numeric_count": len(ref_numbers),
        "pred_numeric_count": len(pred_numbers),
        "numeric_best_absolute_error": (
            min(absolute_errors) if absolute_errors else math.nan
        ),
        "numeric_best_relative_error": (
            min(relative_errors) if relative_errors else math.nan
        ),
        "gold_numbers": [number.raw for number in ref_numbers],
        "pred_numbers": [number.raw for number in pred_numbers],
    }


def evaluate_answer_record(
    *,
    prediction: Any,
    reference: Any,
    question: Any = "",
    context: Any = "",
) -> dict[str, Any]:
    numeric = financial_numeric_metrics(
        prediction,
        reference,
        question=question,
    )
    lexical = lexical_f1(prediction, reference)
    primary_score = (
        numeric["numeric_coverage"]
        if numeric["numeric_applicable"]
        else lexical
    )
    return {
        "lexical_f1": lexical,
        "exact_substring_match": exact_substring_match(prediction, reference),
        "context_support_score": context_support_score(prediction, context),
        "insufficient_evidence": is_insufficient_evidence(prediction),
        "finance_aware_score": float(primary_score),
        **numeric,
    }
