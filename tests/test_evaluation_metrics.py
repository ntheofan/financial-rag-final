import math
import unittest

from scripts.evaluation_metrics import (
    evaluate_answer_record,
    extract_financial_numbers,
    financial_numeric_metrics,
)


class FinanceAwareEvaluationTests(unittest.TestCase):
    def test_millions_suffix_matches_question_inferred_unit(self) -> None:
        result = financial_numeric_metrics(
            "Net working capital is $5,818M.",
            "$5818.00",
            question="What is net working capital? Answer in USD millions.",
        )
        self.assertTrue(result["numeric_applicable"])
        self.assertEqual(result["numeric_match"], 1.0)
        self.assertEqual(result["numeric_coverage"], 1.0)

    def test_percentage_matches_decimal_ratio_with_rounding_tolerance(self) -> None:
        result = financial_numeric_metrics(
            "ROA = -1.53%",
            "-0.02",
            question="What is return on assets (ROA)? Round to two decimals.",
        )
        self.assertEqual(result["numeric_match"], 1.0)

    def test_wrong_financial_value_does_not_match(self) -> None:
        result = financial_numeric_metrics(
            "$465 million",
            "$5818.00",
            question="Answer in USD millions.",
        )
        self.assertEqual(result["numeric_match"], 0.0)
        self.assertEqual(result["numeric_coverage"], 0.0)

    def test_single_gold_scores_final_number_not_intermediate_operand(self) -> None:
        result = financial_numeric_metrics(
            "The table mentions $5,818M, but the final answer is $465M.",
            "$5818.00",
            question="Answer in USD millions.",
        )
        self.assertEqual(result["numeric_match"], 0.0)

    def test_accounting_negative_is_extracted(self) -> None:
        numbers = extract_financial_numbers("Net income was $(546) million.")
        self.assertEqual(len(numbers), 1)
        self.assertEqual(numbers[0].canonical_value, -546_000_000.0)

    def test_calendar_years_are_not_scored_as_answer_quantities(self) -> None:
        numbers = extract_financial_numbers(
            "Margins declined from 36.8% in FY2021 to 34.6% in FY2022."
        )
        self.assertEqual([number.raw for number in numbers], ["36.8%", "34.6%"])

    def test_bare_integer_identifiers_are_not_financial_gold_values(self) -> None:
        result = financial_numeric_metrics(
            "Production will increase for the 737, 777X and 787 aircraft.",
            "Boeing forecasts increases for the 737, 777X and 787 aircraft.",
            question="What production rate changes is Boeing forecasting?",
        )
        self.assertFalse(result["numeric_applicable"])

    def test_non_numeric_reference_uses_lexical_primary_score(self) -> None:
        record = evaluate_answer_record(
            prediction="The shareholder proposal was defeated.",
            reference="The proposal was defeated.",
        )
        self.assertFalse(record["numeric_applicable"])
        self.assertTrue(math.isnan(record["numeric_match"]))
        self.assertEqual(record["finance_aware_score"], record["lexical_f1"])


if __name__ == "__main__":
    unittest.main()
