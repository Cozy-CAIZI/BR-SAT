import unittest

import numpy as np

from br_sat_decision import (
    aggregate_member_probabilities,
    direct_argmax,
    map_three_class_to_binary,
    reactive_score,
    threshold_binary_decision,
)


class DecisionRuleTests(unittest.TestCase):
    def test_probability_aggregation_and_argmax_traceability(self):
        members = np.array(
            [
                [[0.60, 0.25, 0.15], [0.20, 0.45, 0.35]],
                [[0.50, 0.30, 0.20], [0.10, 0.35, 0.55]],
            ]
        )
        mean = aggregate_member_probabilities(members)
        three_class = direct_argmax(mean)
        binary = map_three_class_to_binary(three_class)
        np.testing.assert_array_equal(three_class, [0, 2])
        np.testing.assert_array_equal(binary, [0, 1])
        np.testing.assert_allclose(reactive_score(mean), [0.45, 0.85])

    def test_frozen_binary_threshold(self):
        mean = np.array([[0.40, 0.35, 0.25]])
        # The three-class argmax remains available for traceability, while the
        # primary binary decision uses the development-selected q_R threshold.
        np.testing.assert_array_equal(map_three_class_to_binary(direct_argmax(mean)), [0])
        np.testing.assert_allclose(reactive_score(mean), [0.60])
        np.testing.assert_array_equal(threshold_binary_decision(mean), [1])

    def test_threshold_boundary_is_inclusive(self):
        mean = np.array([[0.428, 0.300, 0.272], [0.429, 0.300, 0.271]])
        np.testing.assert_array_equal(threshold_binary_decision(mean), [1, 0])


if __name__ == "__main__":
    unittest.main()
