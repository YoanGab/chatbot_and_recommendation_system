import unittest

from services.recommendation.similarity import cosine_similarity


class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(
            cosine_similarity([2 / 3, 0, 0, 5 / 3, -7 / 3, 0, 0], [1 / 3, 1 / 3, -2 / 3, 0, 0, 0, 0]),
            0.092,
            places=3
        )
        self.assertAlmostEqual(
            cosine_similarity([2 / 3, 0, 0, 5 / 3, -7 / 3, 0, 0], [0, 0, 0, -5 / 3, 1 / 3, 4 / 3, 0]),
            0.559,
            places=3
        )


if __name__ == '__main__':
    unittest.main()
