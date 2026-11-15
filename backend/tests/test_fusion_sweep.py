"""
Unit Tests for Fusion Sweep Components
=======================================
Tests for:
1. clean_v3:
   - Dates keep their years (e.g. "June 30, 2024", "December 31, 2023", "January 29, 2025")
   - Fiscal-year phrases stripped ("fiscal year 2024", "FY2024", "fiscal 2024", "Form 10-K", preambles)
   - Company name kept (e.g. "Alphabet", "Amazon", "Apple", "Microsoft", "NVIDIA")
2. interleave_4_4:
   - Alternating top results, deduplicating, balancing 4/4 in top 8
3. exact_mcnemar_test:
   - Exact two-sided binomial p-value calculation
"""

import pytest
from backend.eval.fusion_sweep import (
    strip_query_metadata_v3,
    interleave_4_4,
    exact_mcnemar_test,
)


class TestCleanV3:
    def test_dates_keep_their_years(self):
        """Verifies that calendar dates with years (e.g. June 30, 2024) are not stripped."""
        q1 = "At Microsoft, what does the filing say about long-term debt ending June 30, 2024?"
        cleaned1 = strip_query_metadata_v3(q1)
        assert "June 30, 2024" in cleaned1

        q2 = "At Amazon, what were the delivery service expenses for the year ended December 31, 2023?"
        cleaned2 = strip_query_metadata_v3(q2)
        assert "December 31, 2023" in cleaned2

        q3 = "At Amazon, who holds the position of chief executive officer as of January 29, 2025?"
        cleaned3 = strip_query_metadata_v3(q3)
        assert "January 29, 2025" in cleaned3

    def test_fiscal_year_phrases_stripped(self):
        """Verifies that preambles, 10-K references, and fiscal year phrases are removed."""
        q1 = "According to Apple's Form 10-K, what was total revenue in fiscal year 2024?"
        cleaned1 = strip_query_metadata_v3(q1)
        assert "According to" not in cleaned1
        assert "Form 10-K" not in cleaned1
        assert "fiscal year 2024" not in cleaned1
        assert "total revenue" in cleaned1

        q2 = "Apple FY2025 provision for income taxes"
        cleaned2 = strip_query_metadata_v3(q2)
        assert "FY2025" not in cleaned2
        assert "Apple provision for income taxes" in cleaned2

        q3 = "Amazon AWS revenue fiscal year 2024 absolute amount in millions"
        cleaned3 = strip_query_metadata_v3(q3)
        assert "fiscal year 2024" not in cleaned3
        assert "Amazon AWS revenue absolute amount in millions" in cleaned3

        q4 = "What resource price increases are potential factors, per Amazon's fiscal 2024 annual report?"
        cleaned4 = strip_query_metadata_v3(q4)
        assert "annual report" not in cleaned4
        assert "fiscal 2024" not in cleaned4
        assert "What resource price increases are potential factors" in cleaned4

        q5 = "Which executive holds CFO, in Amazon's FY2025 10-K?"
        cleaned5 = strip_query_metadata_v3(q5)
        assert "FY2025" not in cleaned5
        assert "10-K" not in cleaned5
        assert "Which executive holds CFO" in cleaned5

    def test_company_name_kept(self):
        """Verifies that the company name is preserved throughout the query."""
        queries = [
            ("At Alphabet, what is the greenhouse gas target?", "Alphabet"),
            ("At Amazon, how much did AWS add to property?", "Amazon"),
            ("Apple Vision Pro customer satisfaction ratings", "Apple"),
            ("At Microsoft, what consequences could arise from public contracting?", "Microsoft"),
            ("At NVIDIA, compute and Networking segment operating income", "NVIDIA"),
        ]
        for q, comp in queries:
            cleaned = strip_query_metadata_v3(q)
            assert comp in cleaned


class TestInterleave44:
    def test_interleave_no_overlap(self):
        """Verifies round-robin alternating without overlap."""
        list_a = ["a1", "a2", "a3", "a4", "a5"]
        list_b = ["b1", "b2", "b3", "b4", "b5"]
        res = interleave_4_4(list_a, list_b, top_k=8)
        assert res == ["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"]
        assert len(res) == 8

    def test_interleave_with_duplicates(self):
        """Verifies that duplicates are skipped and alternating balance is maintained."""
        list_a = ["c1", "c2", "c3", "c4"]
        list_b = ["c1", "c5", "c6", "c7"]  # c1 is duplicate
        res = interleave_4_4(list_a, list_b, top_k=6)
        # Turn 1 (A): c1
        # Turn 2 (B): c1 seen -> skips to c5
        # Turn 3 (A): c2
        # Turn 4 (B): c6
        # Turn 5 (A): c3
        # Turn 6 (B): c7
        assert res == ["c1", "c5", "c2", "c6", "c3", "c7"]
        assert len(res) == 6
        assert len(set(res)) == 6

    def test_interleave_exhausted_list(self):
        """Verifies graceful handling when one list is shorter than top_k/2."""
        list_a = ["a1"]
        list_b = ["b1", "b2", "b3", "b4"]
        res = interleave_4_4(list_a, list_b, top_k=4)
        assert res == ["a1", "b1", "b2", "b3"]


class TestMcNemarHelper:
    def test_mcnemar_identical(self):
        """When two configs have identical hits, discordant=0 and p=1.0."""
        hits_a = [True, False, True, True]
        hits_b = [True, False, True, True]
        res = exact_mcnemar_test(hits_a, hits_b)
        assert res["both"] == 3
        assert res["neither"] == 1
        assert res["only_a"] == 0
        assert res["only_b"] == 0
        assert res["discordant"] == 0
        assert res["p_value"] == 1.0

    def test_mcnemar_discordant_calculation(self):
        """Verifies exact binomial p-value calculation on known discordant pairs."""
        # 5 only_a, 14 only_b -> p = scipy.stats.binomtest(5, 19, 0.5).pvalue ≈ 0.063568
        hits_a = [True] * 5 + [False] * 14 + [True] * 10 + [False] * 10
        hits_b = [False] * 5 + [True] * 14 + [True] * 10 + [False] * 10
        res = exact_mcnemar_test(hits_a, hits_b)
        assert res["only_a"] == 5
        assert res["only_b"] == 14
        assert res["discordant"] == 19
        assert pytest.approx(res["p_value"], abs=1e-5) == 0.063568

    def test_mcnemar_extreme_discordant(self):
        """When one config gets all 6 discordant hits and other gets 0: p = 2 * (0.5)^6 = 0.03125."""
        hits_a = [True] * 6
        hits_b = [False] * 6
        res = exact_mcnemar_test(hits_a, hits_b)
        assert res["discordant"] == 6
        assert pytest.approx(res["p_value"], abs=1e-5) == 0.03125
