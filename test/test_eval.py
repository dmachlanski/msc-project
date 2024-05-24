import unittest
import numpy as np
import sys
sys.path.append("../")
from utils import eval_peaks, eval_peaks_alt

class TestEvalPeaks(unittest.TestCase):

    def test_eval_peaks_1(self):
        actual = [[10, 20, 30]]
        pred = [[10, 20, 30]]
        result = eval_peaks(actual, pred, 50, 2, 5)
        self.assertEqual(len(result.keys()), 1, "Expected one matching source")
        first = list(result.values())[0]
        self.assertEqual(first['roa'], 1.0, "Expected RoA = 1.0")
        self.assertEqual(first['precision'], 1.0, "Expected Precision = 1.0")
        self.assertEqual(first['recall'], 1.0, "Expected Recall = 1.0")

    def test_eval_peaks_2(self):
        actual = [[10, 20, 30]]
        pred = [[11, 19, 32]]
        result = eval_peaks(actual, pred, 50, 2, 5)
        self.assertEqual(len(result.keys()), 1, "Expected one matching source")
        first = list(result.values())[0]
        self.assertEqual(first['roa'], 1.0, "Expected RoA = 1.0")
        self.assertEqual(first['precision'], 1.0, "Expected Precision = 1.0")
        self.assertEqual(first['recall'], 1.0, "Expected Recall = 1.0")

    def test_eval_peaks_3(self):
        actual = [[100, 200, 300]]
        pred = [[110, 211, 308]]
        result = eval_peaks(actual, pred, 400, 2, 10)
        self.assertEqual(len(result.keys()), 1, "Expected one matching source")
        first = list(result.values())[0]
        self.assertEqual(first['roa'], 1.0, "Expected RoA = 1.0")
        self.assertEqual(first['precision'], 1.0, "Expected Precision = 1.0")
        self.assertEqual(first['recall'], 1.0, "Expected Recall = 1.0")

    def test_eval_peaks_alt_1(self):
        actual = np.array([[10, 20, 30]])
        pred = np.array([[10, 20, 30]])
        result = eval_peaks_alt(actual, pred, 50, 2)
        self.assertEqual(len(result.keys()), 1, "Expected one matching source")
        first = list(result.values())[0]
        self.assertEqual(first['roa'], 1.0, "Expected RoA = 1.0")
        self.assertEqual(first['precision'], 1.0, "Expected Precision = 1.0")
        self.assertEqual(first['recall'], 1.0, "Expected Recall = 1.0")

    def test_eval_peaks_alt_2(self):
        actual = np.array([[10, 20, 30]])
        pred = np.array([[11, 19, 32]])
        result = eval_peaks_alt(actual, pred, 50, 2)
        self.assertEqual(len(result.keys()), 1, "Expected one matching source")
        first = list(result.values())[0]
        self.assertEqual(first['roa'], 1.0, "Expected RoA = 1.0")
        self.assertEqual(first['precision'], 1.0, "Expected Precision = 1.0")
        self.assertEqual(first['recall'], 1.0, "Expected Recall = 1.0")

    def test_eval_peaks_alt_3(self):
        actual = np.array([[100, 200, 300]])
        pred = np.array([[110, 211, 308]])
        result = eval_peaks_alt(actual, pred, 400, 2)
        self.assertEqual(len(result.keys()), 1, "Expected one matching source")
        first = list(result.values())[0]
        self.assertEqual(first['roa'], 1.0, "Expected RoA = 1.0")
        self.assertEqual(first['precision'], 1.0, "Expected Precision = 1.0")
        self.assertEqual(first['recall'], 1.0, "Expected Recall = 1.0")

if __name__ == "__main__":
    unittest.main()