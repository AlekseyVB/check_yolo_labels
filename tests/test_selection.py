import unittest
from check_yolo_labels import evenly_spaced_indices, parse


class SelectionTests(unittest.TestCase):
    def test_dataset_example(self):
        self.assertEqual(
            [i + 1 for i in evenly_spaced_indices(425, 10)],
            [1, 48, 95, 142, 189, 237, 284, 331, 378, 425],
        )

    def test_grid(self):
        for total in range(1, 101):
            for count in range(1, 103):
                with self.subTest(total=total, count=count):
                    indices = evenly_spaced_indices(total, count)
                    self.assertEqual(len(indices), min(total, count))
                    self.assertEqual(len(indices), len(set(indices)))
                    self.assertEqual(indices, sorted(indices))
                    self.assertEqual(indices[0], 0)
                    if len(indices) > 1:
                        self.assertEqual(indices[-1], total - 1)
                        steps = [b - a for a, b in zip(indices, indices[1:])]
                        self.assertLessEqual(max(steps) - min(steps), 1)

    def test_invalid_counts(self):
        for total, count in [(0, 1), (1, 0), (-1, 2), (3, -1)]:
            with self.assertRaises(ValueError):
                evenly_spaced_indices(total, count)


class LabelTests(unittest.TestCase):
    def test_normalized_box(self):
        rows, issues = parse('0 0.5 0.5 0.2 0.4\n')
        self.assertEqual(rows, [(0, 0.5, 0.5, 0.2, 0.4)])
        self.assertEqual(issues, [])

    def test_wrong_task_format(self):
        rows, issues = parse('0 0.1 0.2 0.3 0.4 0.5 0.6\n')
        self.assertEqual(rows, [])
        self.assertTrue(issues)


if __name__ == '__main__':
    unittest.main()
