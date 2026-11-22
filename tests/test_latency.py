import unittest

from src.evaluation.latency import percentile, summarize_latencies


class LatencyTests(unittest.TestCase):
    def test_percentile_interpolates_between_values(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0), 1.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 100), 4.0)

    def test_percentile_of_single_value(self):
        self.assertEqual(percentile([0.5], 99), 0.5)

    def test_summary_reports_milliseconds_and_throughput(self):
        stats = summarize_latencies([0.1, 0.2, 0.3])
        self.assertAlmostEqual(stats["mean_ms"], 200.0)
        self.assertAlmostEqual(stats["median_ms"], 200.0)
        self.assertAlmostEqual(stats["total_s"], 0.6)
        self.assertAlmostEqual(stats["qps"], 5.0)
        self.assertEqual(stats["count"], 3.0)

    def test_median_ignores_a_single_slow_outlier(self):
        stats = summarize_latencies([0.01, 0.01, 0.01, 0.01, 5.0])
        self.assertAlmostEqual(stats["median_ms"], 10.0)
        self.assertGreater(stats["mean_ms"], 900.0)

    def test_warmup_drops_leading_queries(self):
        stats = summarize_latencies([9.0, 0.1, 0.1], warmup=1)
        self.assertEqual(stats["count"], 2.0)
        self.assertAlmostEqual(stats["mean_ms"], 100.0)

    def test_warmup_is_ignored_when_it_would_remove_everything(self):
        stats = summarize_latencies([0.1, 0.2], warmup=5)
        self.assertEqual(stats["count"], 2.0)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            summarize_latencies([])
        with self.assertRaises(ValueError):
            summarize_latencies([0.1], warmup=-1)
        with self.assertRaises(ValueError):
            percentile([], 50)


if __name__ == "__main__":
    unittest.main()
