import unittest

from TTSP.evidence_ledger import normalize_evidence_ledger
from TTSP.prompts import build_evidence_ledger_messages


class EvidenceLedgerTests(unittest.TestCase):
    def test_normalization_enforces_tier_caps(self):
        raw = """## CONFIRMED KNOWLEDGE
1. first
2. second
3. third

## OPEN CONFLICTS
1. alpha vs beta | inspect sign
2. gamma vs delta | inspect corner
"""
        ledger = normalize_evidence_ledger(raw, confirmed_cap=2, conflict_cap=1)
        self.assertIn("1. first", ledger)
        self.assertIn("2. second", ledger)
        self.assertNotIn("third", ledger)
        self.assertIn("alpha vs beta", ledger)
        self.assertNotIn("gamma vs delta", ledger)

    def test_prompt_sorts_traces_by_reliability(self):
        traces = [
            {"texts": ["less reliable"], "reliability_score": -1.0},
            {"texts": ["more reliable"], "reliability_score": -0.1},
        ]
        messages = build_evidence_ledger_messages(
            question="Which?",
            options=["A", "B"],
            image_path="image.png",
            retained_traces=traces,
            max_traces_for_context=1,
        )
        user_text = messages[1]["content"][1]["text"]
        self.assertIn("more reliable", user_text)
        self.assertNotIn("less reliable", user_text)


if __name__ == "__main__":
    unittest.main()
