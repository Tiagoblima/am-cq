import unittest


class QbnnTest(unittest.TestCase):
    test_dic = {(): [('0', 100.0)],
                (0,): [('0', 100.0)],
                (1,): [('0', 100.0)],
                (2,): [('0', 100.0)],
                (0, 1): [('1', 100.0)],
                (0, 2): [('1', 100.0)],
                (1, 2): [('1', 100.0)],
                (0, 1, 2): [('1', 100.0)]}

    def test_output(self, entry, output):
        expected = self.test_dic.get(entry)

        self.assertEqual(expected, output)
