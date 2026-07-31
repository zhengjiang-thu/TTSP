import unittest

from TTSP.voting import compute_voting_results, normalize_answer


class VotingTests(unittest.TestCase):
    def test_normalizes_multiple_choice_letters(self):
        self.assertEqual(normalize_answer(" a. "), "A")

    def test_votes_only_over_supplied_retained_pool(self):
        retained = [
            {"extracted_answer": "A", "reliability_score": -5.0},
            {"extracted_answer": "A", "reliability_score": -5.0},
            {"extracted_answer": "B", "reliability_score": -0.1},
        ]
        result = compute_voting_results(retained, gamma=1.0)
        self.assertEqual(result["TTSP"]["answer"], "B")
        self.assertEqual(result["TTSP"]["num_votes"], 3)

    def test_option_text_maps_to_option_letter(self):
        retained = [
            {"extracted_answer": "Blue", "reliability_score": -0.2},
            {"extracted_answer": "B", "reliability_score": -0.3},
        ]
        result = compute_voting_results(
            retained,
            gamma=1.0,
            options=["Red", "Blue", "Green"],
        )
        self.assertEqual(result["TTSP"]["answer"], "B")


if __name__ == "__main__":
    unittest.main()
